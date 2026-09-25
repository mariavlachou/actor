"""Extract topics from the euaa case-law chunks using Legal-BERT features
guided by the AsyLex annotation labels, then cluster with BERTopic.

Reads a chunks CSV produced by chunk_documents.py (e.g.
euaa_asylum_report_chunks.csv) and the label descriptions in
asylex_labels.txt (CREDIBILITY, DETERMINATION, LEGAL_GROUND, EXPLANATION --
the AsyLex span-annotation categories). For each chunk, Legal-BERT
(nlpaueb/legal-bert-base-uncased) embeds both the chunk text and each label's
description; the chunk's feature vector is its mean-pooled Legal-BERT
embedding concatenated with its cosine similarity to each of the four label
embeddings, so the label vocabulary directly shapes the feature space BERTopic
clusters on (not just the raw embedding). BERTopic (UMAP -> HDBSCAN ->
c-TF-IDF) is then run on these precomputed features, and each topic is named
with a local Ollama model, mirroring extract_topics.py's pipeline.

Two DataFrames come out of this:
  - the per-chunk assignment (topic id/keywords/name/probability plus each
    chunk's per-label similarity scores and dominant label), written to
    `output_csv`.
  - the reduced topic-level table (one row per topic, in the same shape as
    extract_topics.py's *_topic_names.csv: topic_id, topic_name,
    topic_keywords, plus a chunk count), written to `topics_output_csv` and
    returned by `run()`.

Usage as a library:

    from extract_topics_legalbert import LegalBertTopicConfig, LegalBertTopicExtractor

    config = LegalBertTopicConfig(input_csv="euaa_asylum_report_chunks.csv")
    reduced_topics_df = LegalBertTopicExtractor(config).run()   # also writes both CSVs

Usage from the command line:

    python3 extract_topics_legalbert.py euaa_asylum_report_chunks.csv
    python3 extract_topics_legalbert.py euaa_asylum_report_chunks.csv \
        --labels-file asylex_labels.txt --min-cluster-size 5

Requires: transformers, torch, bertopic, umap-learn, hdbscan, pandas, requests
(pip install transformers torch bertopic pandas), plus a running Ollama server
with the chosen model pulled (`ollama pull hf.co/bartowski/google_gemma-3-4b-it-GGUF:Q5_K_S`).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from generate_queries import call_ollama


def parse_labels_file(path: str) -> Dict[str, str]:
    """Parse "LABEL: description" blocks out of asylex_labels.txt into {label: description}."""
    text = open(path, encoding="utf-8").read()
    matches = re.findall(r"^([A-Z_]+):\s*(.+?)(?=\n[A-Z_]+:|\Z)", text, flags=re.DOTALL | re.MULTILINE)
    if not matches:
        raise ValueError(f"No 'LABEL: description' entries found in {path}")
    return {label.strip(): " ".join(desc.split()) for label, desc in matches}


@dataclass
class LegalBertTopicConfig:
    """What to cluster, which models to use, and how each pipeline stage behaves."""

    input_csv: str
    text_column: str = "chunk_text"
    labels_file: str = "prompts/asylex_labels.txt"
    output_csv: Optional[str] = None  # default: "<input>_legalbert_topics.csv"
    topics_output_csv: Optional[str] = None  # default: "<input>_legalbert_topic_names.csv"

    # Legal-BERT feature extraction
    legalbert_model_name: str = "nlpaueb/legal-bert-base-uncased"
    max_seq_length: int = 512
    batch_size: int = 16
    device: Optional[str] = None  # default: "cuda" if available, else "cpu"

    # UMAP (non-linear dimensionality reduction). Lower than extract_topics.py's
    # defaults: unlike sentence-transformers query embeddings, Legal-BERT's raw
    # mean-pooled embeddings are anisotropic (most chunks point in a similar
    # direction), so a smaller neighbourhood is needed to resolve local structure.
    umap_n_neighbors: int = 5
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    # HDBSCAN (hierarchical clustering)
    hdbscan_min_cluster_size: int = 5
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
    num_representative_docs: int = 5  # example chunks shown to the LLM per topic
    name_outliers: bool = False  # if False, topic -1 is labelled "Outliers" without an LLM call


def build_topic_naming_prompt(keywords: List[str], representative_docs: List[str]) -> str:
    keyword_line = ", ".join(keywords)
    docs_block = "\n".join(f"- {d}" for d in representative_docs)
    return (
        "You are naming a topic cluster produced by a topic-modeling pipeline over asylum case-law text chunks.\n"
        f"Top keywords for this cluster: {keyword_line}\n"
        "Representative chunks in this cluster:\n"
        f"{docs_block}\n\n"
        "Give a short, descriptive name for this topic (3-6 words, title case).\n"
        "Only output the name, without any additional text."
    )


class LegalBertFeatureExtractor:
    """Mean-pools Legal-BERT hidden states into one embedding per input text."""

    def __init__(self, model_name: str, max_seq_length: int, batch_size: int, device: Optional[str]):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        all_embeddings = []

        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)

                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state  # (batch, seq_len, hidden)
                mask = encoded["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)

                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_pooled = (summed / counts).cpu().numpy()
                all_embeddings.append(mean_pooled)

        return np.concatenate(all_embeddings, axis=0)


class LegalBertTopicExtractor:
    """Runs the Legal-BERT feature extraction -> BERTopic -> LLM-naming pipeline over a chunks CSV."""

    def __init__(self, config: LegalBertTopicConfig):
        self.config = config

    def _load_chunks(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.input_csv)
        if self.config.text_column not in df.columns:
            raise ValueError(
                f"Column {self.config.text_column!r} not found; available columns: {list(df.columns)}"
            )
        df = df[df[self.config.text_column].notna()].reset_index(drop=True)
        return df

    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, str]]:
        """Legal-BERT chunk embeddings, per-chunk label-similarity scores, and the parsed labels."""
        cfg = self.config
        labels = parse_labels_file(cfg.labels_file)

        extractor = LegalBertFeatureExtractor(
            cfg.legalbert_model_name, cfg.max_seq_length, cfg.batch_size, cfg.device
        )

        texts = df[cfg.text_column].astype(str).tolist()
        chunk_embeddings = extractor.encode(texts)
        label_embeddings = extractor.encode(list(labels.values()))

        chunk_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True).clip(min=1e-9)
        label_norm = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True).clip(min=1e-9)
        # Apple Accelerate's BLAS occasionally raises spurious divide/overflow
        # FP warnings on this matmul shape without ever producing a bad value
        # (checked: chunk_norm/label_norm/similarities are always finite) --
        # suppressed here rather than left to alarm on every run.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarities = chunk_norm @ label_norm.T  # (n_chunks, n_labels) cosine similarity

        label_scores = {label: similarities[:, i] for i, label in enumerate(labels)}

        # The clustering feature vector: the chunk's own (L2-normalised) Legal-BERT
        # embedding, with its similarity to each AsyLex label appended -- so the
        # label vocabulary in asylex_labels.txt directly shapes what BERTopic groups on.
        features = np.concatenate([chunk_norm, similarities], axis=1)
        return features, label_scores, labels

    def _build_model(self):
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from umap import UMAP

        cfg = self.config

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
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            top_n_words=cfg.top_n_words,
            calculate_probabilities=True,
            verbose=True,
        )

    def _fit(self) -> Tuple[pd.DataFrame, Any, Dict[str, str]]:
        """Extract features, then cluster. Returns the per-chunk DataFrame, the fitted model, and the labels."""
        df = self._load_chunks()
        texts: List[str] = df[self.config.text_column].astype(str).tolist()

        features, label_scores, labels = self._extract_features(df)

        topic_model = self._build_model()
        topics, probs = topic_model.fit_transform(texts, embeddings=features)

        topic_info = topic_model.get_topic_info().set_index("Topic")
        keyword_lookup = topic_info["Representation"].apply(lambda words: ", ".join(words)).to_dict()

        df = df.copy()
        df.insert(0, "topic_id", topics)
        df["topic_keywords"] = df["topic_id"].map(keyword_lookup)
        df["topic_probability"] = [p.max() if hasattr(p, "max") else p for p in probs] if probs is not None else None

        for label, scores in label_scores.items():
            df[f"{label.lower()}_score"] = scores
        df["dominant_label"] = pd.DataFrame(label_scores).idxmax(axis=1)

        return df, topic_model, labels

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
        """One row per unique topic id: its LLM name, keywords, and how many chunks it holds."""
        topic_info = topic_model.get_topic_info().set_index("Topic")
        counts = df["topic_id"].value_counts()

        rows = [
            {
                "topic_id": topic_id,
                "topic_name": topic_names.get(topic_id, ""),
                "topic_keywords": ", ".join(topic_info.loc[topic_id, "Representation"]),
                "num_chunks": int(counts.get(topic_id, 0)),
            }
            for topic_id in topic_info.index
        ]
        return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """Extract, name, and write both CSVs. Returns the reduced topic-level DataFrame."""
        df, topic_model, _labels = self._fit()
        topic_names = self._name_topics(topic_model)
        df["topic_name"] = df["topic_id"].map(topic_names)

        per_chunk_output = self.config.output_csv or self._default_output_path()
        df.to_csv(per_chunk_output, index=False)

        reduced_df = self._build_reduced_topics_df(df, topic_model, topic_names)
        topics_output = self.config.topics_output_csv or self._default_topics_output_path()
        reduced_df.to_csv(topics_output, index=False)

        return reduced_df

    def _default_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_legalbert_topics{ext or '.csv'}"

    def _default_topics_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_legalbert_topic_names{ext or '.csv'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = LegalBertTopicConfig(input_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Chunks CSV to extract topics from (must have a chunk_text-style text column)")
    parser.add_argument("--text-column", default=defaults.text_column, help="Column holding the chunk text")
    parser.add_argument("--labels-file", default=defaults.labels_file, help="Path to the 'LABEL: description' file (e.g. asylex_labels.txt)")
    parser.add_argument("--legalbert-model", default=defaults.legalbert_model_name, help="Legal-BERT model name")
    parser.add_argument("--min-cluster-size", type=int, default=defaults.hdbscan_min_cluster_size, help="HDBSCAN min_cluster_size")
    parser.add_argument("--n-components", type=int, default=defaults.umap_n_components, help="UMAP output dimensionality")
    parser.add_argument("--top-n-words", type=int, default=defaults.top_n_words, help="Keywords to extract per topic via c-TF-IDF")
    parser.add_argument("--llm-model", default=defaults.llm_model, help="Ollama model used to name each topic")
    parser.add_argument("--ollama-host", default=defaults.ollama_host, help="Ollama server base URL")
    parser.add_argument("--output", default=None, help="Per-chunk output CSV path (default: '<input>_legalbert_topics.csv')")
    parser.add_argument("--topics-output", default=None, help="Reduced topic-level output CSV path (default: '<input>_legalbert_topic_names.csv')")
    args = parser.parse_args()

    config = LegalBertTopicConfig(
        input_csv=args.input_csv,
        text_column=args.text_column,
        labels_file=args.labels_file,
        legalbert_model_name=args.legalbert_model,
        hdbscan_min_cluster_size=args.min_cluster_size,
        umap_n_components=args.n_components,
        top_n_words=args.top_n_words,
        llm_model=args.llm_model,
        ollama_host=args.ollama_host,
        output_csv=args.output,
        topics_output_csv=args.topics_output,
    )
    extractor = LegalBertTopicExtractor(config)
    reduced_df = extractor.run()
    print(f"Extracted {len(reduced_df)} topics -> {config.topics_output_csv or extractor._default_topics_output_path()}")
    print(f"Per-chunk assignments -> {config.output_csv or extractor._default_output_path()}")


if __name__ == "__main__":
    main()
