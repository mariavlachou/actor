"""Extract topics from a generated-queries CSV using the BERTopic pipeline,
then name each topic with a local Ollama LLM.

Reads a queries CSV produced by generate_queries.py (e.g.
fln_praksis_2026_queries.csv), embeds each query with a sentence-transformers
model, reduces the embedding dimensionality with UMAP, clusters the reduced
embeddings into topics with HDBSCAN, and labels each topic with its top
TF-IDF-weighted keywords (BERTopic's class-based c-TF-IDF). Each topic's
keywords and representative queries are then handed to a local Ollama model,
which writes a short human-readable name for the cluster.

Two DataFrames come out of this:
  - the full per-query correspondence (one row per input query, with its
    assigned topic id/keywords/name/probability), written to `output_csv`.
  - the reduced topic-level table (one row per unique topic id, with its
    LLM-generated name, keywords and query count), written to
    `topics_output_csv` and returned by `run()`.

Usage as a library:

    from extract_topics import TopicExtractionConfig, QueryTopicExtractor

    config = TopicExtractionConfig(input_csv="fln_praksis_2026_queries.csv")
    reduced_topics_df = QueryTopicExtractor(config).run()   # also writes both CSVs

Usage from the command line:

    python3 extract_topics.py fln_praksis_2026_queries.csv
    python3 extract_topics.py euaa_asylum_report_queries.csv \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --llm-model gemma3:4b --min-cluster-size 5 --output euaa_topics.csv

Requires: bertopic, umap-learn, hdbscan, sentence-transformers, pandas, requests
(pip install bertopic pandas), plus a running Ollama server with the chosen
model pulled (`ollama pull hf.co/bartowski/google_gemma-3-4b-it-GGUF:Q5_K_S`).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from generate_queries import call_ollama


@dataclass
class TopicExtractionConfig:
    """What to cluster, which models to use, and how each pipeline stage behaves."""

    input_csv: str
    query_column: str = "generated_query"
    output_csv: Optional[str] = None  # default: "<input>_topics.csv"
    topics_output_csv: Optional[str] = None  # default: "<input>_topic_names.csv"

    # Embedding
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # UMAP (non-linear dimensionality reduction)
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    # HDBSCAN (hierarchical clustering)
    hdbscan_min_cluster_size: int = 10
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"

    # c-TF-IDF keyword extraction per topic
    top_n_words: int = 10

    random_state: int = 42

    # LLM-based topic naming (via a local Ollama server)
    llm_model: str = "hf.co/bartowski/google_gemma-3-4b-it-GGUF:Q5_K_S"
    ollama_host: str = "http://localhost:11434"
    llm_temperature: float = 0.3
    llm_request_timeout: int = 120
    num_representative_docs: int = 5  # example queries shown to the LLM per topic
    name_outliers: bool = False  # if False, topic -1 is labelled "Outliers" without an LLM call


def build_topic_naming_prompt(keywords: List[str], representative_docs: List[str]) -> str:
    keyword_line = ", ".join(keywords)
    docs_block = "\n".join(f"- {d}" for d in representative_docs)
    return (
        "You are naming a topic cluster produced by a topic-modeling pipeline over search queries.\n"
        f"Top keywords for this cluster: {keyword_line}\n"
        "Representative queries in this cluster:\n"
        f"{docs_block}\n\n"
        "Give a short, descriptive name for this topic (3-6 words, title case).\n"
        "Only output the name, without any additional text."
    )


class QueryTopicExtractor:
    """Runs the embed -> reduce -> cluster -> c-TF-IDF-label -> LLM-name pipeline over a queries CSV."""

    def __init__(self, config: TopicExtractionConfig):
        self.config = config

    def _load_queries(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.input_csv)
        if self.config.query_column not in df.columns:
            raise ValueError(
                f"Column {self.config.query_column!r} not found; available columns: {list(df.columns)}"
            )
        df = df[df[self.config.query_column].notna()].reset_index(drop=True)
        return df

    def _build_model(self):
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from umap import UMAP

        cfg = self.config

        embedding_model = SentenceTransformer(cfg.embedding_model_name)

        umap_model = UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            n_components=cfg.umap_n_components,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=cfg.random_state,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            metric=cfg.hdbscan_metric,
            cluster_selection_method=cfg.hdbscan_cluster_selection_method,
            prediction_data=True,
        )

        ctfidf_model = ClassTfidfTransformer()  # TF-IDF weighting for per-topic keywords

        return BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            top_n_words=cfg.top_n_words,
            calculate_probabilities=True,
            verbose=True,
        )

    def _fit(self) -> Tuple[pd.DataFrame, Any]:
        """Embed, reduce, and cluster the queries. Returns the per-query DataFrame and the fitted model."""
        df = self._load_queries()
        queries: List[str] = df[self.config.query_column].astype(str).tolist()

        topic_model = self._build_model()
        topics, probs = topic_model.fit_transform(queries)

        topic_info = topic_model.get_topic_info().set_index("Topic")
        keyword_lookup = topic_info["Representation"].apply(lambda words: ", ".join(words)).to_dict()
        label_lookup = topic_info["Name"].to_dict()

        df = df.copy()
        df.insert(0, "topic_id", topics)
        df["topic_label"] = df["topic_id"].map(label_lookup)
        df["topic_keywords"] = df["topic_id"].map(keyword_lookup)
        df["topic_probability"] = [p.max() if hasattr(p, "max") else p for p in probs] if probs is not None else None

        return df, topic_model

    def extract(self) -> pd.DataFrame:
        """Per-query topic assignment only (no LLM naming). Kept for lighter-weight use."""
        df, _ = self._fit()
        return df

    def _name_topics(self, topic_model) -> Dict[int, str]:
        """Ask the local Ollama model for a short name per topic, keyed by topic id."""
        cfg = self.config
        topic_ids = topic_model.get_topic_info()["Topic"].tolist()
        names: Dict[int, str] = {}

        for topic_id in topic_ids:
            if topic_id == -1 and not cfg.name_outliers:
                names[topic_id] = "Outliers"
                continue

            keywords = [word for word, _ in topic_model.get_topic(topic_id)][: cfg.top_n_words]
            representative_docs = (topic_model.get_representative_docs(topic_id) or [])[: cfg.num_representative_docs]
            prompt = build_topic_naming_prompt(keywords, representative_docs)

            try:
                name = call_ollama(cfg.llm_model, prompt, cfg.ollama_host, cfg.llm_temperature, cfg.llm_request_timeout)
            except Exception as exc:  # network error, timeout, model not pulled, etc.
                print(f"warning: topic naming failed for topic {topic_id}: {exc}", file=sys.stderr)
                name = ""
            names[topic_id] = name.strip().strip('"')

        return names

    def _build_reduced_topics_df(self, df: pd.DataFrame, topic_model, topic_names: Dict[int, str]) -> pd.DataFrame:
        """One row per unique topic id: its LLM name, keywords, and how many queries it holds."""
        topic_info = topic_model.get_topic_info().set_index("Topic")
        counts = df["topic_id"].value_counts()

        rows = [
            {
                "topic_id": topic_id,
                "topic_name": topic_names.get(topic_id, ""),
                "topic_keywords": ", ".join(topic_info.loc[topic_id, "Representation"]),
                "num_queries": int(counts.get(topic_id, 0)),
            }
            for topic_id in topic_info.index
        ]
        return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """Extract, name, and write both CSVs. Returns the reduced topic-level DataFrame."""
        df, topic_model = self._fit()
        topic_names = self._name_topics(topic_model)
        df["topic_name"] = df["topic_id"].map(topic_names)

        per_query_output = self.config.output_csv or self._default_output_path()
        df.to_csv(per_query_output, index=False)

        reduced_df = self._build_reduced_topics_df(df, topic_model, topic_names)
        topics_output = self.config.topics_output_csv or self._default_topics_output_path()
        reduced_df.to_csv(topics_output, index=False)

        return reduced_df

    def _default_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_topics{ext or '.csv'}"

    def _default_topics_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_topic_names{ext or '.csv'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = TopicExtractionConfig(input_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Queries CSV to extract topics from (must have a generated_query-style text column)")
    parser.add_argument("--query-column", default=defaults.query_column, help="Column holding the query text")
    parser.add_argument("--embedding-model", default=defaults.embedding_model_name, help="sentence-transformers model name")
    parser.add_argument("--min-cluster-size", type=int, default=defaults.hdbscan_min_cluster_size, help="HDBSCAN min_cluster_size")
    parser.add_argument("--n-components", type=int, default=defaults.umap_n_components, help="UMAP output dimensionality")
    parser.add_argument("--top-n-words", type=int, default=defaults.top_n_words, help="Keywords to extract per topic via c-TF-IDF")
    parser.add_argument("--llm-model", default=defaults.llm_model, help="Ollama model used to name each topic")
    parser.add_argument("--ollama-host", default=defaults.ollama_host, help="Ollama server base URL")
    parser.add_argument("--output", default=None, help="Per-query output CSV path (default: '<input>_topics.csv')")
    parser.add_argument("--topics-output", default=None, help="Reduced topic-level output CSV path (default: '<input>_topic_names.csv')")
    args = parser.parse_args()

    config = TopicExtractionConfig(
        input_csv=args.input_csv,
        query_column=args.query_column,
        embedding_model_name=args.embedding_model,
        hdbscan_min_cluster_size=args.min_cluster_size,
        umap_n_components=args.n_components,
        top_n_words=args.top_n_words,
        llm_model=args.llm_model,
        ollama_host=args.ollama_host,
        output_csv=args.output,
        topics_output_csv=args.topics_output,
    )
    extractor = QueryTopicExtractor(config)
    reduced_df = extractor.run()
    print(f"Extracted {len(reduced_df)} topics -> {config.topics_output_csv or extractor._default_topics_output_path()}")
    print(f"Per-query assignments -> {config.output_csv or extractor._default_output_path()}")


if __name__ == "__main__":
    main()
