"""Sample rows from a Hugging Face dataset into a CSV file.

The dataset URL, subset (config), split, sample size, and text column are all
parameters, so the same script works for other Hugging Face datasets -- not
just the default target.

Default target: https://huggingface.co/datasets/clairebarale/AsyLex
  subset (config): raw_documents
  split:           train
  samples:         200
  text column:     txt

Usage as a library:

    from hf_dataset_sampler import HFDatasetConfig, HFDatasetSampler

    config = HFDatasetConfig(
        dataset_url="https://huggingface.co/datasets/clairebarale/AsyLex",
        subset="raw_documents",
        split="train",
        num_samples=200,
        text_column="txt",
    )
    rows = HFDatasetSampler(config).run("asylex_sample.csv")

Usage from the command line:

    python3 hf_dataset_sampler.py
    python3 hf_dataset_sampler.py \
        --dataset-url https://huggingface.co/datasets/some/other-dataset \
        --subset some_subset --split validation --num-samples 50 \
        --text-column text --output other_sample.csv

Requires the `datasets` library:  pip install datasets
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class HFDatasetConfig:
    """Everything needed to pick a slice of a Hugging Face dataset and sample it."""

    dataset_url: str = "https://huggingface.co/datasets/clairebarale/AsyLex"
    subset: Optional[str] = "raw_documents"  # the dataset's "config name"; None if it has no subsets
    split: str = "train"
    num_samples: int = 200
    text_column: str = "txt"
    id_column: Optional[str] = None  # auto-detected (__key__/id/ID/...) when left unset
    seed: Optional[int] = 42  # set to None for a non-reproducible random sample


def repo_id_from_url(dataset_url: str) -> str:
    """Turn a huggingface.co dataset page URL into a `datasets.load_dataset` repo id."""
    match = re.search(r"huggingface\.co/datasets/([^/?#]+/[^/?#]+)", dataset_url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[\w.\-]+/[\w.\-]+", dataset_url):
        return dataset_url  # already an "owner/name" repo id
    raise ValueError(f"Could not extract a Hugging Face dataset repo id from {dataset_url!r}")


class HFDatasetSampler:
    """Loads a dataset split, draws a random sample, and writes it to CSV."""

    def __init__(self, config: HFDatasetConfig):
        self.config = config

    def load_dataset(self):
        repo_id = repo_id_from_url(self.config.dataset_url)
        return load_dataset(repo_id, name=self.config.subset, split=self.config.split)

    def _resolve_id_column(self, column_names: List[str]) -> Optional[str]:
        if self.config.id_column is not None:
            if self.config.id_column not in column_names:
                raise ValueError(f"id_column {self.config.id_column!r} not found; available columns: {column_names}")
            return self.config.id_column
        for candidate in ("__key__", "id", "ID", "Id", "doc_id", "case_id"):
            if candidate in column_names:
                return candidate
        return None

    def sample(self, dataset=None) -> List[Dict[str, Any]]:
        cfg = self.config
        if dataset is None:
            dataset = self.load_dataset()

        if cfg.text_column not in dataset.column_names:
            raise ValueError(
                f"Text column {cfg.text_column!r} not found; available columns: {dataset.column_names}"
            )

        n = min(cfg.num_samples, len(dataset))
        rng = random.Random(cfg.seed)
        indices = rng.sample(range(len(dataset)), n)

        id_column = self._resolve_id_column(dataset.column_names)

        rows: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices):
            record = dataset[idx]
            item_id = str(record[id_column]) if id_column else str(idx)
            row = {"item_id": item_id}
            row.update(record)
            rows.append(row)
        return rows

    @staticmethod
    def save_to_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
        fieldnames = list(rows[0].keys()) if rows else ["item_id"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run(self, output_path: str) -> List[Dict[str, Any]]:
        """Sample and write CSV in one call. Returns the sampled rows too."""
        rows = self.sample()
        self.save_to_csv(rows, output_path)
        return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = HFDatasetConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-url", default=defaults.dataset_url, help="Hugging Face dataset page URL or 'owner/name' repo id")
    parser.add_argument("--subset", default=defaults.subset, help="Dataset subset/config name (omit or pass '' for datasets without subsets)")
    parser.add_argument("--split", default=defaults.split, help="Dataset split, e.g. train/validation/test")
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples, help="Number of rows to sample")
    parser.add_argument("--text-column", default=defaults.text_column, help="Name of the column holding the item text")
    parser.add_argument("--id-column", default=defaults.id_column, help="Column to use as item_id (auto-detected if omitted)")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed for reproducible sampling")
    parser.add_argument("--output", default="hf_dataset_sample.csv", help="Output CSV path")
    args = parser.parse_args()

    config = HFDatasetConfig(
        dataset_url=args.dataset_url,
        subset=args.subset or None,
        split=args.split,
        num_samples=args.num_samples,
        text_column=args.text_column,
        id_column=args.id_column,
        seed=args.seed,
    )
    rows = HFDatasetSampler(config).run(args.output)
    print(f"Sampled {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
