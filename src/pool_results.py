"""Pool top-ranked results from multiple retrieval runs into one deduplicated CSV.

Takes the per-query result CSVs produced by pt_retrieval.py (BM25/SPLADE,
with or without monoT5 rerank) and pt_dense_retrieval.py (E5/Qwen3) for one
dataset, keeps only their shared columns (qid, query, docno, text, rank,
score), concatenates them, and removes duplicates so each retrieved
query-document pair appears only once -- a standard TREC-style "pool" of
candidates.

"Duplicate" here means the same (qid, docno) pair showing up in more than one
retrieval run's results (e.g. the same chunk retrieved by both BM25 and
SPLADE for the same query) -- not a byte-identical row, since rank/score are
on different scales per method and essentially never coincide across
retrievers. The first file to contribute a given (qid, docno) pair "wins"
that row's rank/score in the pool.

Which files go into the pool is chosen by --dataset: a short label (euaa,
fln, or asylex) that expands to that dataset's topics-file stem, then globs
every CSV starting with that stem and keeps only the ones that actually look
like retrieval results (have docno/rank/score columns) -- so it automatically
picks up whichever retriever runs you've actually produced for that dataset,
and skips the plain topics file itself. Override with --results-glob for a
custom pattern (e.g. to pool only a specific topic sample), or --results to
name the exact files.

Usage as a library:

    from pool_results import PoolingConfig, ResultPooler

    config = PoolingConfig(dataset="euaa")
    df = ResultPooler(config).run()   # also writes the output CSV

Usage from the command line:

    python3 pool_results.py --dataset euaa
    python3 pool_results.py --dataset fln --results-glob "fln_praksis_2026_queries_topic_names_qid_query_sample30_*.csv"
    python3 pool_results.py --dataset asylex --results run1.csv run2.csv run3.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

POOL_COLUMNS = ["qid", "query", "docno", "text", "rank", "score"]

# Maps a short --dataset label to the stem shared by that dataset's topics
# file and every retrieval result file derived from it (see topic_to_queries.py).
DATASET_TOPIC_STEMS = {
    "euaa": "data/euaa_asylum_report_queries_topic_names_qid_query",
    "fln": "data/fln_praksis_2026_queries_topic_names_qid_query",
    "asylex": "data/asylex_raw_documents_sample_queries_topic_names_qid_query",
}


@dataclass
class PoolingConfig:
    """Which dataset's retrieval results to pool, and how to find/dedupe them."""

    dataset: str  # a DATASET_TOPIC_STEMS key, or any custom stem/label
    results_glob: Optional[str] = None  # default: "<dataset stem>*.csv"
    result_files: Optional[List[str]] = None  # explicit file list; overrides results_glob
    dedupe_on: Sequence[str] = field(default_factory=lambda: ("qid", "docno"))
    output_csv: Optional[str] = None  # default: "<dataset stem>_pool.csv"


class ResultPooler:
    """Discovers a dataset's retrieval result files, merges, and deduplicates them."""

    def __init__(self, config: PoolingConfig):
        self.config = config

    def _dataset_stem(self) -> str:
        return DATASET_TOPIC_STEMS.get(self.config.dataset, self.config.dataset)

    def _discover_files(self) -> List[str]:
        cfg = self.config
        if cfg.result_files:
            return list(cfg.result_files)

        pattern = cfg.results_glob or f"{self._dataset_stem()}*.csv"
        result_files = []
        for path in sorted(glob.glob(pattern)):
            if os.path.basename(path).endswith("_pool.csv"):
                continue  # never feed a previous pool run's own output back in as a source
            try:
                header = set(pd.read_csv(path, nrows=0).columns)
            except Exception:
                continue
            if {"docno", "rank", "score"}.issubset(header):
                result_files.append(path)  # skip the bare topics file / anything non-retrieval
        return result_files

    def pool(self) -> pd.DataFrame:
        cfg = self.config
        files = self._discover_files()
        if not files:
            pattern = cfg.results_glob or f"{self._dataset_stem()}*.csv"
            raise ValueError(f"No retrieval result files found for dataset {cfg.dataset!r} (pattern: {pattern!r})")

        frames = []
        for path in files:
            df = pd.read_csv(path, dtype={"qid": str, "docno": str})
            missing = [c for c in POOL_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"{path} missing column(s) {missing}; available columns: {list(df.columns)}")
            frames.append(df[POOL_COLUMNS])
            print(f"  + {path}: {len(df)} rows")

        merged = pd.concat(frames, ignore_index=True)
        pooled = merged.drop_duplicates(subset=list(cfg.dedupe_on), keep="first").reset_index(drop=True)
        print(f"Pooled {len(merged)} rows from {len(files)} file(s) -> {len(pooled)} unique {tuple(cfg.dedupe_on)} pairs")
        return pooled

    def run(self) -> pd.DataFrame:
        """Pool, deduplicate, and write CSV in one call. Returns the DataFrame too."""
        df = self.pool()
        df.to_csv(self.config.output_csv or self._default_output_path(), index=False)
        return df

    def _default_output_path(self) -> str:
        return f"{self._dataset_stem()}_pool.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help=f"Dataset label ({', '.join(DATASET_TOPIC_STEMS)}) or a custom filename stem")
    parser.add_argument("--results-glob", default=None, help="Custom glob pattern for result files (default: '<dataset stem>*.csv')")
    parser.add_argument("--results", nargs="+", default=None, help="Explicit list of result CSV paths, instead of globbing")
    parser.add_argument("--dedupe-on", nargs="+", default=["qid", "docno"], help="Columns that define a duplicate (default: qid docno)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<dataset stem>_pool.csv')")
    args = parser.parse_args()

    config = PoolingConfig(
        dataset=args.dataset,
        results_glob=args.results_glob,
        result_files=args.results,
        dedupe_on=tuple(args.dedupe_on),
        output_csv=args.output,
    )
    pooler = ResultPooler(config)
    df = pooler.run()
    output_path = config.output_csv or pooler._default_output_path()
    print(f"Saved pool of {len(df)} rows ({df['qid'].nunique()} queries) -> {output_path}")


if __name__ == "__main__":
    main()
