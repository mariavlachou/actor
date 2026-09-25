"""Select topic_id/topic_name from a topic-level CSV and rename them to qid/query.

Reads a reduced topic-level CSV produced by extract_topics.py (e.g.
fln_praksis_2026_queries_topic_names.csv), keeps only its id and name
columns, renames them to `qid` and `query`, and writes the result as a new
CSV -- handy for treating each topic as a query in a downstream retrieval
evaluation.

Usage as a library:

    from topic_to_queries import TopicSubsetConfig, TopicQuerySubsetter

    config = TopicSubsetConfig(input_csv="fln_praksis_2026_queries_topic_names.csv")
    df = TopicQuerySubsetter(config).run()   # also writes the output CSV

Usage from the command line:

    python3 topic_to_queries.py fln_praksis_2026_queries_topic_names.csv
    python3 topic_to_queries.py euaa_asylum_report_queries_topic_names.csv --output euaa_qid_query.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TopicSubsetConfig:
    """Which columns to pull out of the topic CSV, and what to call the output."""

    input_csv: str
    id_column: str = "topic_id"
    name_column: str = "topic_name"
    output_csv: Optional[str] = None  # default: "<input>_qid_query.csv"


class TopicQuerySubsetter:
    """Reads a topic-level CSV and reduces it to a qid/query pair table."""

    def __init__(self, config: TopicSubsetConfig):
        self.config = config

    def subset(self) -> pd.DataFrame:
        cfg = self.config
        df = pd.read_csv(cfg.input_csv)

        missing = [col for col in (cfg.id_column, cfg.name_column) if col not in df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not found; available columns: {list(df.columns)}")

        return df[[cfg.id_column, cfg.name_column]].rename(
            columns={cfg.id_column: "qid", cfg.name_column: "query"}
        )

    def run(self) -> pd.DataFrame:
        """Subset and write CSV in one call. Returns the DataFrame too."""
        df = self.subset()
        df.to_csv(self.config.output_csv or self._default_output_path(), index=False)
        return df

    def _default_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_qid_query{ext or '.csv'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = TopicSubsetConfig(input_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Topic-level CSV to pull qid/query from")
    parser.add_argument("--id-column", default=defaults.id_column, help="Column to rename to 'qid'")
    parser.add_argument("--name-column", default=defaults.name_column, help="Column to rename to 'query'")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<input>_qid_query.csv')")
    args = parser.parse_args()

    config = TopicSubsetConfig(
        input_csv=args.input_csv,
        id_column=args.id_column,
        name_column=args.name_column,
        output_csv=args.output,
    )
    subsetter = TopicQuerySubsetter(config)
    df = subsetter.run()
    output_path = config.output_csv or subsetter._default_output_path()
    print(f"Wrote {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
