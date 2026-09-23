"""Compute rank metrics for a retrieval run using PyTerrier.

Reads a retrieval result CSV (produced by pt_retrieval.py or
pt_dense_retrieval.py; columns include qid, docno, score) and its
corresponding relevance judgments from a *_labels.csv file (produced by
judge_pool.py; columns: qid, docno, label), evaluates the run with
pt.Evaluate, and returns/saves: MAP@100, NDCG@10, MRR@10, P@5, R@5.

The dataset and retrieval method are both arguments: --dataset picks the
topics-file stem (a DATASET_TOPIC_STEMS label, or any custom stem), and
--retriever picks which result file to evaluate (e.g. "bm25", "bm25_monot5",
"splade", "e5", "qwen3") -- together they resolve the default results path
"<dataset stem>_<retriever>.csv" and labels path "<dataset stem>_pool_labels.csv",
the same naming convention pt_retrieval.py / pool_results.py / judge_pool.py
already use. Pass --results-csv / --labels-csv directly to evaluate a run
that doesn't follow that convention (e.g. a topic sample).

Usage as a library:

    from evaluate_run import EvaluationConfig, RunEvaluator

    config = EvaluationConfig(dataset="euaa", retriever="bm25")
    metrics_df = RunEvaluator(config).run()   # also writes the output CSV

Usage from the command line:

    python3 evaluate_run.py --dataset euaa --retriever bm25
    python3 evaluate_run.py --dataset fln --retriever splade \
        --results-csv fln_praksis_2026_queries_topic_names_qid_query_sample30_splade.csv \
        --labels-csv fln_praksis_2026_queries_topic_names_qid_query_sample30_pool_labels.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyterrier as pt

# Maps a short --dataset label to the stem shared by that dataset's topics,
# pool, and result files (see topic_to_queries.py / pool_results.py).
DATASET_TOPIC_STEMS = {
    "euaa": "euaa_asylum_report_queries_topic_names_qid_query",
    "fln": "fln_praksis_2026_queries_topic_names_qid_query",
    "asylex": "asylex_raw_documents_sample_queries_topic_names_qid_query",
}


@dataclass
class EvaluationConfig:
    """Which run to evaluate, against which qrels."""

    dataset: str  # a DATASET_TOPIC_STEMS key, or a custom stem
    retriever: str  # e.g. "bm25", "bm25_monot5", "splade", "e5", "qwen3"
    results_csv: Optional[str] = None  # default: "<dataset stem>_<retriever>.csv"
    labels_csv: Optional[str] = None  # default: "<dataset stem>_pool_labels.csv"
    output_csv: Optional[str] = None  # default: "<dataset stem>_<retriever>_metrics.csv"


class RunEvaluator:
    """Loads a run + its qrels and computes MAP@100, NDCG@10, MRR@10, P@5, R@5 with PyTerrier."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        if not pt.started():
            pt.init()

    def _dataset_stem(self) -> str:
        return DATASET_TOPIC_STEMS.get(self.config.dataset, self.config.dataset)

    def _default_results_path(self) -> str:
        return f"{self._dataset_stem()}_{self.config.retriever}.csv"

    def _default_labels_path(self) -> str:
        return f"{self._dataset_stem()}_pool_labels.csv"

    def _default_output_path(self) -> str:
        return f"{self._dataset_stem()}_{self.config.retriever}_metrics.csv"

    def _load_results(self) -> pd.DataFrame:
        path = self.config.results_csv or self._default_results_path()
        df = pd.read_csv(path, dtype={"qid": str, "docno": str})
        missing = [c for c in ("qid", "docno", "score") if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing column(s) {missing}; available columns: {list(df.columns)}")
        return df

    def _load_labels(self) -> pd.DataFrame:
        path = self.config.labels_csv or self._default_labels_path()
        df = pd.read_csv(path, dtype={"qid": str, "docno": str})
        missing = [c for c in ("qid", "docno", "label") if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing column(s) {missing}; available columns: {list(df.columns)}")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df

    def evaluate(self) -> pd.DataFrame:
        from pyterrier.measures import AP, RR, P, R, nDCG

        results = self._load_results()
        labels = self._load_labels()

        scores = pt.Evaluate(results, labels, metrics=[AP @ 100, nDCG @ 10, RR @ 10, P @ 5, R @ 5])

        row = {"dataset": self.config.dataset, "retriever": self.config.retriever}
        row.update(scores)
        return pd.DataFrame([row])

    def run(self) -> pd.DataFrame:
        """Evaluate and write CSV in one call. Returns the one-row metrics DataFrame too."""
        df = self.evaluate()
        df.to_csv(self.config.output_csv or self._default_output_path(), index=False)
        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help=f"Dataset label ({', '.join(DATASET_TOPIC_STEMS)}) or a custom filename stem")
    parser.add_argument("--retriever", required=True, help="Retrieval method to evaluate, e.g. bm25, bm25_monot5, splade, e5, qwen3")
    parser.add_argument("--results-csv", default=None, help="Results CSV path (default: '<dataset stem>_<retriever>.csv')")
    parser.add_argument("--labels-csv", default=None, help="Labels/qrels CSV path (default: '<dataset stem>_pool_labels.csv')")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<dataset stem>_<retriever>_metrics.csv')")
    args = parser.parse_args()

    config = EvaluationConfig(
        dataset=args.dataset,
        retriever=args.retriever,
        results_csv=args.results_csv,
        labels_csv=args.labels_csv,
        output_csv=args.output,
    )
    evaluator = RunEvaluator(config)
    df = evaluator.run()
    output_path = config.output_csv or evaluator._default_output_path()
    print(df.to_string(index=False))
    print(f"Saved metrics -> {output_path}")


if __name__ == "__main__":
    main()
