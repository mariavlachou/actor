"""Test run of chunk_documents.py against one of the three source CSVs.

Defaults to asylex_raw_documents_sample.csv (text column "txt"); the input
CSV, its text column, and the output path can all be overridden, either from
the command line (--input, --text-column, --output) or programmatically
(pass `input_csv=`, `text_column=`, `output_path=` to main()).
"""

import argparse

from chunk_documents import ChunkingConfig, DocumentChunker

DEFAULT_INPUT_CSV = "asylex_raw_documents_sample.csv"
DEFAULT_TEXT_COLUMN = "txt"


def main(
    input_csv: str = DEFAULT_INPUT_CSV,
    text_column: str = DEFAULT_TEXT_COLUMN,
    output_path: str = None,
) -> None:
    config = ChunkingConfig(input_csv=input_csv, text_column=text_column, output_csv=output_path)
    chunker = DocumentChunker(config)
    rows = chunker.run()

    resolved_output = output_path or chunker._default_output_path()
    num_docs = len({r["doc_id"] for r in rows})
    print(f"Chunked {num_docs} documents into {len(rows)} chunks -> {resolved_output}")
    for row in rows[:3]:
        print(row["chunk_id"], "|", row["chunk_index"], "/", row["total_chunks"], "|", row["chunk_text"][:80].replace("\n", " "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Source document-level CSV to chunk")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Column holding the case text")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<input>_chunks.csv')")
    args = parser.parse_args()
    main(input_csv=args.input, text_column=args.text_column, output_path=args.output)
