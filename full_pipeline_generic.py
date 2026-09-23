"""Full pipeline example: scraping -> evaluation metrics (any dataset), as a script.

Plain-.py equivalent of full_pipeline_generic.ipynb: the same 11-stage
pipeline (source -> chunk -> generate queries -> extract topics -> reduce
topics -> chunks-to-docs -> sample -> retrieve -> pool -> judge -> evaluate),
generalised to any of this project's three datasets via one `--dataset` flag,
calling the same real project modules the notebook does.

Why sourcing (step 1) is a branch, not a single call: each dataset was
acquired a genuinely different way in this project --
  - "fln":    scraper.py's plain GET-based WebsiteScraper (fln.dk)
  - "euaa":   browser_scraper.py's Playwright-driven BrowserWebsiteScraper
              (fln.dk-style scraping doesn't work there)
  - "asylex": hf_dataset_sampler.py pulling a Hugging Face dataset, not a
              scrape at all

Steps 2-11 are dataset-agnostic already -- every underlying script takes its
input file as an argument -- so they just read whichever config the chosen
dataset resolves to.

Every stage is idempotent: it loads the existing output file if one is
already on disk, and only computes it otherwise. Several stages are slow
LLM/embedding jobs (per-chunk query generation, topic naming, LLM-as-judge
labelling can each take from minutes to hours on a full dataset), so this
script is safe to re-run -- it skips straight past anything already done.

Usage as a library:

    from full_pipeline_generic import run_pipeline

    metrics_df = run_pipeline(dataset_key="fln", sample_size=30, sample_seed=42)

Usage from the command line:

    python3 full_pipeline_generic.py --dataset fln
    python3 full_pipeline_generic.py --dataset euaa --sample-size 50

Requires a Python distribution where PyTerrier's embedded JVM actually
starts (Anaconda's, on this machine) for the retrieve/evaluate stages.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

SAMPLE_SIZE_DEFAULT = 30
SAMPLE_SEED_DEFAULT = 42
RETRIEVERS = ["bm25", "bm25_monot5", "splade", "e5", "qwen3"]

# Everything dataset-specific lives here. Every stage below only ever reads
# from a resolved DatasetPaths built from one of these entries -- to run this
# script for a different dataset, pick a different --dataset, nothing else
# needs to change.
DATASET_REGISTRY = {
    "fln": dict(stem="fln_praksis_2026", source_method="scraper", text_column="description"),
    "euaa": dict(
        stem="euaa_asylum_report",
        source_method="browser_scraper",
        text_column="description",
        filter_label="Input Provider",
        filter_value="EUAA Asylum Report",
    ),
    "asylex": dict(stem="asylex_raw_documents_sample", source_method="hf_dataset", text_column="txt"),
}


@dataclass
class DatasetPaths:
    """Every filename this pipeline reads/writes for one dataset + sample size, all derived
    from `stem` using the same "<stem>_..." convention used throughout the project."""

    dataset_key: str
    stem: str
    source_method: str
    text_column: str
    sample_size: int
    filter_label: Optional[str] = None
    filter_value: Optional[str] = None

    @property
    def scrape_csv(self) -> str:
        return f"{self.stem}.csv"

    @property
    def chunks_csv(self) -> str:
        return f"{self.stem}_chunks.csv"

    @property
    def queries_csv(self) -> str:
        return f"{self.stem}_queries.csv"

    @property
    def per_query_topics_csv(self) -> str:
        return f"{self.stem}_queries_topics.csv"

    @property
    def topic_names_csv(self) -> str:
        return f"{self.stem}_queries_topic_names.csv"

    @property
    def topics_qid_query_csv(self) -> str:
        return f"{self.stem}_queries_topic_names_qid_query.csv"

    @property
    def chunks_docs_csv(self) -> str:
        return f"{self.stem}_chunks_docs.csv"

    @property
    def sample_topics_csv(self) -> str:
        return f"{self.stem}_queries_topic_names_qid_query_sample{self.sample_size}.csv"

    @property
    def sample_stem(self) -> str:
        return os.path.splitext(self.sample_topics_csv)[0]

    @property
    def pool_csv(self) -> str:
        return f"{self.sample_stem}_pool.csv"

    @property
    def labels_csv(self) -> str:
        return f"{self.sample_stem}_pool_labels.csv"

    @classmethod
    def for_dataset(cls, dataset_key: str, sample_size: int = SAMPLE_SIZE_DEFAULT) -> "DatasetPaths":
        cfg = DATASET_REGISTRY[dataset_key]
        return cls(dataset_key=dataset_key, sample_size=sample_size, **cfg)


def step_done(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def source_data(paths: DatasetPaths) -> pd.DataFrame:
    """Step 1: acquire the raw dataset (branches on source_method -- see module docstring)."""
    if not step_done(paths.scrape_csv):
        print(f"Sourcing {paths.dataset_key} data ({paths.source_method}) -> {paths.scrape_csv} ...")
        if paths.source_method == "scraper":
            from scraper import FLN_PRAKSIS, WebsiteScraper

            WebsiteScraper(FLN_PRAKSIS).run(paths.scrape_csv)
        elif paths.source_method == "browser_scraper":
            from browser_scraper import EUAA_CASELAW, BrowserWebsiteScraper

            BrowserWebsiteScraper(EUAA_CASELAW).run(
                paths.scrape_csv, filter_label=paths.filter_label, filter_value=paths.filter_value
            )
        elif paths.source_method == "hf_dataset":
            from hf_dataset_sampler import HFDatasetConfig, HFDatasetSampler

            HFDatasetSampler(HFDatasetConfig()).run(paths.scrape_csv)
        else:
            raise ValueError(f"Unknown source_method {paths.source_method!r}")
    else:
        print(f"Already sourced -> {paths.scrape_csv} (skipping)")
    return pd.read_csv(paths.scrape_csv)


def chunk_data(paths: DatasetPaths) -> pd.DataFrame:
    """Step 2: split each document into ~400-char chunks at natural boundaries."""
    from chunk_documents import ChunkingConfig, DocumentChunker

    if not step_done(paths.chunks_csv):
        config = ChunkingConfig(input_csv=paths.scrape_csv, text_column=paths.text_column, output_csv=paths.chunks_csv)
        DocumentChunker(config).run()
    else:
        print(f"Already chunked -> {paths.chunks_csv} (skipping)")
    return pd.read_csv(paths.chunks_csv)


def generate_queries_step(paths: DatasetPaths) -> pd.DataFrame:
    """Step 3: one few-shot-generated query per chunk, via a local Ollama LLM. Slow on a
    full dataset (~85 minutes for fln's 5,844 chunks) -- skipped if already done."""
    from generate_queries import QueryGenConfig, QueryGenerator

    if not step_done(paths.queries_csv):
        print("Generating one query per chunk via Ollama (this can take a while)...")
        config = QueryGenConfig(input_csv=paths.chunks_csv, output_csv=paths.queries_csv)
        QueryGenerator(config).run()
    else:
        print(f"Already generated -> {paths.queries_csv} (skipping)")
    return pd.read_csv(paths.queries_csv)


def extract_topics_step(paths: DatasetPaths) -> pd.DataFrame:
    """Step 4: embed queries, cluster with BERTopic, name each topic with a local LLM."""
    from extract_topics import QueryTopicExtractor, TopicExtractionConfig

    if not (step_done(paths.topic_names_csv) and step_done(paths.per_query_topics_csv)):
        print("Embedding, clustering, and naming topics (this can take a minute or two)...")
        config = TopicExtractionConfig(
            input_csv=paths.queries_csv, output_csv=paths.per_query_topics_csv, topics_output_csv=paths.topic_names_csv
        )
        QueryTopicExtractor(config).run()
    else:
        print(f"Already extracted -> {paths.topic_names_csv} (skipping)")
    return pd.read_csv(paths.topic_names_csv)


def reduce_topics_step(paths: DatasetPaths) -> pd.DataFrame:
    """Step 5: reduce the topic table to a plain qid/query pair table."""
    from topic_to_queries import TopicQuerySubsetter, TopicSubsetConfig

    if not step_done(paths.topics_qid_query_csv):
        config = TopicSubsetConfig(input_csv=paths.topic_names_csv, output_csv=paths.topics_qid_query_csv)
        TopicQuerySubsetter(config).run()
    else:
        print(f"Already reduced -> {paths.topics_qid_query_csv} (skipping)")
    return pd.read_csv(paths.topics_qid_query_csv, dtype={"qid": str})


def chunks_to_docs_step(paths: DatasetPaths) -> pd.DataFrame:
    """Step 6: chunk_id/chunk_text -> docno/text, the columns PyTerrier indexing expects."""
    from chunks_to_docs import ChunkDocsConfig, ChunkDocsConverter

    if not step_done(paths.chunks_docs_csv):
        config = ChunkDocsConfig(input_csv=paths.chunks_csv, output_csv=paths.chunks_docs_csv)
        ChunkDocsConverter(config).run()
    else:
        print(f"Already converted -> {paths.chunks_docs_csv} (skipping)")
    return pd.read_csv(paths.chunks_docs_csv)


def sample_topics_step(paths: DatasetPaths, all_topics: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Step 7: a fixed, seeded random sample of the full topic set."""
    if not step_done(paths.sample_topics_csv):
        n = min(paths.sample_size, len(all_topics))
        sample = all_topics.sample(n=n, random_state=seed).reset_index(drop=True)
        sample.to_csv(paths.sample_topics_csv, index=False)
    else:
        print(f"Already sampled -> {paths.sample_topics_csv} (skipping)")
    return pd.read_csv(paths.sample_topics_csv, dtype={"qid": str})


def retrieve_step(paths: DatasetPaths) -> Dict[str, str]:
    """Step 8: BM25, BM25+monoT5, SPLADE (sparse) and E5, Qwen3 (dense) retrieval runs.
    The Danish stemmer is auto-detected from the docs filename, so no dataset branching
    is needed here."""
    from pt_dense_retrieval import DenseRetrievalConfig, DenseRetrievalPipeline
    from pt_retrieval import PyTerrierRetrievalPipeline, RetrievalConfig

    result_paths: Dict[str, str] = {}

    for retriever in ["bm25", "bm25_monot5", "splade"]:
        rerank = retriever.endswith("_monot5")
        base_retriever = "bm25" if rerank else retriever
        out_path = f"{paths.sample_stem}_{retriever}.csv"
        if not step_done(out_path):
            print(f"Running {retriever} ...")
            config = RetrievalConfig(
                docs_csv=paths.chunks_docs_csv,
                topics_csv=paths.sample_topics_csv,
                retriever=base_retriever,
                rerank_monot5=rerank,
                output_csv=out_path,
            )
            PyTerrierRetrievalPipeline(config).run()
        else:
            print(f"Already ran {retriever} -> {out_path} (skipping)")
        result_paths[retriever] = out_path

    for retriever in ["e5", "qwen3"]:
        out_path = f"{paths.sample_stem}_{retriever}.csv"
        if not step_done(out_path):
            print(f"Running {retriever} ...")
            config = DenseRetrievalConfig(
                docs_csv=paths.chunks_docs_csv, topics_csv=paths.sample_topics_csv, retriever=retriever, output_csv=out_path
            )
            DenseRetrievalPipeline(config).run()
        else:
            print(f"Already ran {retriever} -> {out_path} (skipping)")
        result_paths[retriever] = out_path

    return result_paths


def pool_step(paths: DatasetPaths) -> pd.DataFrame:
    """Step 9: union every method's results, deduplicated by (qid, docno)."""
    from pool_results import PoolingConfig, ResultPooler

    if not step_done(paths.pool_csv):
        config = PoolingConfig(
            dataset=paths.dataset_key, results_glob=f"{paths.sample_stem}_*.csv", output_csv=paths.pool_csv
        )
        ResultPooler(config).run()
    else:
        print(f"Already pooled -> {paths.pool_csv} (skipping)")
    return pd.read_csv(paths.pool_csv, dtype={"qid": str, "docno": str})


def judge_step(paths: DatasetPaths) -> pd.DataFrame:
    """Step 10: LLM-as-judge relevance labelling of the pool, via a local Ollama model.
    The slowest stage on a large pool (hours) -- skipped whenever a labels file already
    exists. A retrieval method added after judging still gets evaluated against those
    existing labels in step 11 rather than triggering a full re-judge: an unjudged
    document is conventionally treated as not relevant in IR evaluation, which is
    standard practice, not a shortcut -- it just means a method excluded from the
    original pool can score lower than it "really" would under a fresh pool+judge pass."""
    from judge_pool import JudgeConfig, QrelsJudge

    if not step_done(paths.labels_csv):
        print("Judging the pool with a local LLM (this can take hours on a large pool)...")
        config = JudgeConfig(input_csv=paths.pool_csv, output_csv=paths.labels_csv)
        QrelsJudge(config).run()
    else:
        print(f"Already judged -> {paths.labels_csv} (skipping; see note above)")
    return pd.read_csv(paths.labels_csv, dtype={"qid": str, "docno": str}).dropna(subset=["label"])


def evaluate_step(paths: DatasetPaths, result_paths: Dict[str, str]) -> pd.DataFrame:
    """Step 11: MAP@100, NDCG@10, MRR@10, P@5, R@5 for every retrieval method."""
    from evaluate_run import EvaluationConfig, RunEvaluator

    metrics_rows = []
    for retriever, results_path in result_paths.items():
        config = EvaluationConfig(
            dataset=paths.dataset_key,
            retriever=retriever,
            results_csv=results_path,
            labels_csv=paths.labels_csv,
            output_csv=f"{paths.sample_stem}_{retriever}_metrics.csv",
        )
        metrics_rows.append(RunEvaluator(config).run())

    metrics_df = pd.concat(metrics_rows, ignore_index=True).sort_values("AP@100", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(f"{paths.sample_stem}_all_methods_metrics.csv", index=False)
    return metrics_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(
    dataset_key: str = "fln", sample_size: int = SAMPLE_SIZE_DEFAULT, sample_seed: int = SAMPLE_SEED_DEFAULT
) -> pd.DataFrame:
    """Run every stage for one dataset and return the final metrics DataFrame."""
    paths = DatasetPaths.for_dataset(dataset_key, sample_size)
    print(f"Dataset: {dataset_key}  (stem: {paths.stem})")

    source_data(paths)
    chunk_data(paths)
    generate_queries_step(paths)
    extract_topics_step(paths)
    all_topics = reduce_topics_step(paths)
    chunks_to_docs_step(paths)
    sample_topics_step(paths, all_topics, sample_seed)
    result_paths = retrieve_step(paths)
    pool_step(paths)
    judge_step(paths)
    metrics_df = evaluate_step(paths, result_paths)

    print(f"\nFinal metrics -> {paths.sample_stem}_all_methods_metrics.csv")
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY), default="fln", help="Which dataset to run the pipeline for")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE_DEFAULT, help="Number of topics to sample for retrieval/evaluation")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED_DEFAULT, help="Random seed for the topic sample")
    args = parser.parse_args()

    metrics_df = run_pipeline(dataset_key=args.dataset, sample_size=args.sample_size, sample_seed=args.sample_seed)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
