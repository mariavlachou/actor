"""Run retrieval -> pooling -> LLM judging -> evaluation for an existing topic table.

Picks up the pipeline where extract_topics.py / extract_topics_legalbert.py
(or any script producing a topic_id/topic_name topic-level CSV) leaves off:
reduces the topic table to a qid/query pair per topic (topic_to_queries.py),
runs BM25, BM25+monoT5, SPLADE, E5 and Qwen3 retrieval against an existing
*_chunks_docs.csv index (pt_retrieval.py / pt_dense_retrieval.py -- reusing
any index already built for that docs file), pools every method's results
(pool_results.py), judges the pool with a local LLM (judge_pool.py), and
evaluates every method against those judgments (evaluate_run.py), writing one
combined metrics table -- the same shape full_pipeline_generic.py's
evaluate_step produces, just decoupled from its fixed dataset registry so it
can run for an alternative topic-extraction method (e.g. Legal-BERT-derived
topics) over the same document collection.

Usage as a library:

    from run_topic_retrieval_eval import TopicRetrievalEvalConfig, TopicRetrievalEvalPipeline

    config = TopicRetrievalEvalConfig(
        topic_names_csv="euaa_asylum_report_chunks_legalbert_topic_names.csv",
        docs_csv="euaa_asylum_report_chunks_docs.csv",
        dataset_label="euaa_legalbert",
    )
    metrics_df = TopicRetrievalEvalPipeline(config).run()

Usage from the command line:

    python3 run_topic_retrieval_eval.py \
        --topic-names-csv euaa_asylum_report_chunks_legalbert_topic_names.csv \
        --docs-csv euaa_asylum_report_chunks_docs.csv \
        --dataset-label euaa_legalbert

Requires a Python distribution where PyTerrier's embedded JVM actually starts
(Anaconda's, on this machine) plus a running Ollama server for judging.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_RETRIEVERS = ["bm25", "bm25_monot5", "splade", "e5", "qwen3"]


@dataclass
class TopicRetrievalEvalConfig:
    """Which topic table + document collection to run retrieval/eval over, and with what."""

    topic_names_csv: str  # topic_id/topic_name/topic_keywords table (any *_topic_names.csv-shaped file)
    docs_csv: str  # a *_chunks_docs.csv file: columns docno, text
    dataset_label: str  # short label for output filenames + the "dataset" column in the metrics table

    retrievers: List[str] = field(default_factory=lambda: list(DEFAULT_RETRIEVERS))
    judge_model: str = "llama3.1:8b"

    def step_done(self, path: str) -> bool:
        return os.path.exists(path) and os.path.getsize(path) > 0


class TopicRetrievalEvalPipeline:
    """Reduce -> retrieve -> pool -> judge -> evaluate, all keyed off one topic table."""

    def __init__(self, config: TopicRetrievalEvalConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Paths (all derived from the topics_qid_query stem, same convention
    # pt_retrieval.py / pool_results.py / judge_pool.py already use).
    # ------------------------------------------------------------------
    @property
    def topics_qid_query_csv(self) -> str:
        stem, ext = os.path.splitext(self.config.topic_names_csv)
        return f"{stem}_qid_query{ext or '.csv'}"

    @property
    def stem(self) -> str:
        return os.path.splitext(self.topics_qid_query_csv)[0]

    @property
    def pool_csv(self) -> str:
        return f"{self.stem}_pool.csv"

    @property
    def labels_csv(self) -> str:
        return f"{self.stem}_pool_labels.csv"

    @property
    def all_methods_metrics_csv(self) -> str:
        return f"{self.stem}_all_methods_metrics.csv"

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def reduce_topics(self) -> pd.DataFrame:
        from topic_to_queries import TopicQuerySubsetter, TopicSubsetConfig

        if not self.config.step_done(self.topics_qid_query_csv):
            config = TopicSubsetConfig(input_csv=self.config.topic_names_csv, output_csv=self.topics_qid_query_csv)
            TopicQuerySubsetter(config).run()
        else:
            print(f"Already reduced -> {self.topics_qid_query_csv} (skipping)")
        return pd.read_csv(self.topics_qid_query_csv, dtype={"qid": str})

    def retrieve(self) -> Dict[str, str]:
        from pt_dense_retrieval import DenseRetrievalConfig, DenseRetrievalPipeline
        from pt_retrieval import PyTerrierRetrievalPipeline, RetrievalConfig

        result_paths: Dict[str, str] = {}

        for retriever in self.config.retrievers:
            if retriever in ("e5", "qwen3"):
                out_path = f"{self.stem}_{retriever}.csv"
                if not self.config.step_done(out_path):
                    print(f"Running {retriever} ...")
                    dense_config = DenseRetrievalConfig(
                        docs_csv=self.config.docs_csv,
                        topics_csv=self.topics_qid_query_csv,
                        retriever=retriever,
                        output_csv=out_path,
                    )
                    DenseRetrievalPipeline(dense_config).run()
                else:
                    print(f"Already ran {retriever} -> {out_path} (skipping)")
                result_paths[retriever] = out_path
            else:
                rerank = retriever.endswith("_monot5")
                base_retriever = retriever[: -len("_monot5")] if rerank else retriever
                out_path = f"{self.stem}_{retriever}.csv"
                if not self.config.step_done(out_path):
                    print(f"Running {retriever} ...")
                    sparse_config = RetrievalConfig(
                        docs_csv=self.config.docs_csv,
                        topics_csv=self.topics_qid_query_csv,
                        retriever=base_retriever,
                        rerank_monot5=rerank,
                        output_csv=out_path,
                    )
                    PyTerrierRetrievalPipeline(sparse_config).run()
                else:
                    print(f"Already ran {retriever} -> {out_path} (skipping)")
                result_paths[retriever] = out_path

        return result_paths

    def pool(self) -> pd.DataFrame:
        from pool_results import PoolingConfig, ResultPooler

        if not self.config.step_done(self.pool_csv):
            config = PoolingConfig(
                dataset=self.config.dataset_label, results_glob=f"{self.stem}_*.csv", output_csv=self.pool_csv
            )
            ResultPooler(config).run()
        else:
            print(f"Already pooled -> {self.pool_csv} (skipping)")
        return pd.read_csv(self.pool_csv, dtype={"qid": str, "docno": str})

    def judge(self) -> pd.DataFrame:
        from judge_pool import JudgeConfig, QrelsJudge

        if not self.config.step_done(self.labels_csv):
            print("Judging the pool with a local LLM (this can take a while)...")
            config = JudgeConfig(input_csv=self.pool_csv, model=self.config.judge_model, output_csv=self.labels_csv)
            QrelsJudge(config).run()
        else:
            print(f"Already judged -> {self.labels_csv} (skipping)")
        return pd.read_csv(self.labels_csv, dtype={"qid": str, "docno": str}).dropna(subset=["label"])

    def evaluate(self, result_paths: Dict[str, str]) -> pd.DataFrame:
        from evaluate_run import EvaluationConfig, RunEvaluator

        metrics_rows = []
        for retriever, results_path in result_paths.items():
            config = EvaluationConfig(
                dataset=self.config.dataset_label,
                retriever=retriever,
                results_csv=results_path,
                labels_csv=self.labels_csv,
                output_csv=f"{self.stem}_{retriever}_metrics.csv",
            )
            metrics_rows.append(RunEvaluator(config).run())

        metrics_df = pd.concat(metrics_rows, ignore_index=True).sort_values("AP@100", ascending=False).reset_index(drop=True)
        metrics_df.to_csv(self.all_methods_metrics_csv, index=False)
        return metrics_df

    def run(self) -> pd.DataFrame:
        """Run every stage and return the final combined metrics DataFrame."""
        self.reduce_topics()
        result_paths = self.retrieve()
        self.pool()
        self.judge()
        metrics_df = self.evaluate(result_paths)
        print(f"\nFinal metrics -> {self.all_methods_metrics_csv}")
        return metrics_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = TopicRetrievalEvalConfig(topic_names_csv="", docs_csv="", dataset_label="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic-names-csv", required=True, help="Topic-level CSV (topic_id/topic_name/topic_keywords)")
    parser.add_argument("--docs-csv", required=True, help="*_chunks_docs.csv file to retrieve against (columns: docno, text)")
    parser.add_argument("--dataset-label", required=True, help="Short label for output filenames and the metrics table's 'dataset' column")
    parser.add_argument("--retrievers", nargs="+", default=defaults.retrievers, help="Retrieval methods to run/evaluate")
    parser.add_argument("--judge-model", default=defaults.judge_model, help="Ollama model used for LLM-as-judge relevance labelling")
    args = parser.parse_args()

    config = TopicRetrievalEvalConfig(
        topic_names_csv=args.topic_names_csv,
        docs_csv=args.docs_csv,
        dataset_label=args.dataset_label,
        retrievers=args.retrievers,
        judge_model=args.judge_model,
    )
    metrics_df = TopicRetrievalEvalPipeline(config).run()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
