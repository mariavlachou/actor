"""Select chunk_id/chunk_text from a chunks CSV and rename them to docno/text.

Reads a chunk-level CSV produced by chunk_documents.py (e.g.
asylex_raw_documents_sample_chunks.csv), keeps only its id and text columns,
renames them to `docno` and `text` (with `docno` coerced to string), and
writes the result as a new CSV -- the column names PyTerrier-style indexing
and retrieval tools expect for a document collection.

Usage as a library:

    from chunks_to_docs import ChunkDocsConfig, ChunkDocsConverter

    config = ChunkDocsConfig(input_csv="asylex_raw_documents_sample_chunks.csv")
    df = ChunkDocsConverter(config).run()   # also writes the output CSV

Usage from the command line:

    python3 chunks_to_docs.py asylex_raw_documents_sample_chunks.csv
    python3 chunks_to_docs.py fln_praksis_2026_chunks.csv --output fln_docs.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ChunkDocsConfig:
    """Which columns to pull out of the chunks CSV, and what to call the output."""

    input_csv: str
    id_column: str = "chunk_id"
    text_column: str = "chunk_text"
    output_csv: Optional[str] = None  # default: "<input>_docs.csv"


class ChunkDocsConverter:
    """Reads a chunk-level CSV and reduces it to a docno/text pair table."""

    def __init__(self, config: ChunkDocsConfig):
        self.config = config

    def convert(self) -> pd.DataFrame:
        cfg = self.config
        df = pd.read_csv(cfg.input_csv)

        missing = [col for col in (cfg.id_column, cfg.text_column) if col not in df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not found; available columns: {list(df.columns)}")

        subset = df[[cfg.id_column, cfg.text_column]].rename(
            columns={cfg.id_column: "docno", cfg.text_column: "text"}
        )
        subset["docno"] = subset["docno"].astype(str)
        return subset

    def run(self) -> pd.DataFrame:
        """Convert and write CSV in one call. Returns the DataFrame too."""
        df = self.convert()
        df.to_csv(self.config.output_csv or self._default_output_path(), index=False)
        return df

    def _default_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_docs{ext or '.csv'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = ChunkDocsConfig(input_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Chunks CSV to pull docno/text from")
    parser.add_argument("--id-column", default=defaults.id_column, help="Column to rename to 'docno'")
    parser.add_argument("--text-column", default=defaults.text_column, help="Column to rename to 'text'")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<input>_docs.csv')")
    args = parser.parse_args()

    config = ChunkDocsConfig(
        input_csv=args.input_csv,
        id_column=args.id_column,
        text_column=args.text_column,
        output_csv=args.output,
    )
    converter = ChunkDocsConverter(config)
    df = converter.run()
    output_path = config.output_csv or converter._default_output_path()
    print(f"Wrote {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
