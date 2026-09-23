"""Test run of hf_dataset_sampler.py with the default specifications.

Default target: https://huggingface.co/datasets/clairebarale/AsyLex
  subset: raw_documents, split: train, num_samples: 200, text_column: txt
"""

from hf_dataset_sampler import HFDatasetConfig, HFDatasetSampler

OUTPUT_PATH = "asylex_raw_documents_sample.csv"


def main() -> None:
    config = HFDatasetConfig()
    rows = HFDatasetSampler(config).run(OUTPUT_PATH)
    print(f"Sampled {len(rows)} rows -> {OUTPUT_PATH}")
    for row in rows[:3]:
        print(row["item_id"], "|", row["txt"][:80].replace("\n", " "))


if __name__ == "__main__":
    main()
