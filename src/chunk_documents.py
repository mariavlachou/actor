"""Chunk the text column of one of the scraped/sampled asylum-case CSVs into a
chunk-level CSV, preserving the document <-> chunk mapping and every other
original column.

Works with any of: fln_praksis_2026.csv, euaa_asylum_report.csv,
asylex_raw_documents_sample.csv (or any similarly-shaped CSV) -- both the
input file and its text column ("txt" for the AsyLex sample, "description"
for the fln.dk/EUAA scrapes) are given as arguments rather than hardcoded.

Chunking splits text at its natural snippet boundaries rather than at a fixed
character offset: paragraphs first (blank-line breaks), falling back to
sentence boundaries for any paragraph longer than the target chunk size, and
only hard-splitting mid-sentence as a last resort for a single pathologically
long sentence. Paragraphs/sentences are then greedily packed into chunks up
to `max_chunk_chars`, and a trailing chunk shorter than `min_chunk_chars` is
merged into the previous one so documents don't end in a tiny fragment.

The default `max_chunk_chars` (400) targets a couple of sentences per chunk,
capping out at roughly a very small paragraph; raise it for coarser,
paragraph- or section-sized chunks.

Usage as a library:

    from chunk_documents import ChunkingConfig, DocumentChunker

    config = ChunkingConfig(input_csv="asylex_raw_documents_sample.csv", text_column="txt")
    rows = DocumentChunker(config).run()   # also writes the output CSV

Usage from the command line:

    python3 chunk_documents.py asylex_raw_documents_sample.csv --text-column txt
    python3 chunk_documents.py fln_praksis_2026.csv --text-column description \
        --max-chunk-chars 400 --output fln_praksis_2026_chunks.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

csv.field_size_limit(10_000_000)

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
AUTO_DETECT_TEXT_COLUMNS = ("txt", "description")


@dataclass
class ChunkingConfig:
    """What to chunk, and how big a "natural" chunk is allowed to get."""

    input_csv: str
    text_column: Optional[str] = None  # auto-detected ("txt" then "description") if omitted
    id_column: str = "item_id"
    output_csv: Optional[str] = None  # default: "<input>_chunks.csv"
    max_chunk_chars: int = 400  # roughly a couple of sentences, or a very small paragraph
    min_chunk_chars: int = 80  # a trailing chunk shorter than this is merged into the previous one


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def split_sentences(paragraph: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def chunk_text(text: str, max_chars: int, min_chars: int) -> List[str]:
    """Split `text` into chunks at paragraph/sentence boundaries, each up to ~max_chars."""
    text = text.strip()
    if not text:
        return []

    # Pass 1: break into paragraph-sized (or smaller) units, splitting any
    # over-long paragraph at sentence boundaries.
    units: List[str] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue

        buffer = ""
        for sentence in split_sentences(paragraph):
            if len(sentence) > max_chars:
                if buffer:
                    units.append(buffer)
                    buffer = ""
                # A single sentence longer than the target size: hard-split as a last resort.
                for i in range(0, len(sentence), max_chars):
                    units.append(sentence[i : i + max_chars])
                continue
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) > max_chars and buffer:
                units.append(buffer)
                buffer = sentence
            else:
                buffer = candidate
        if buffer:
            units.append(buffer)

    # Pass 2: greedily pack units back into chunks up to max_chars.
    chunks: List[str] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer}\n\n{unit}".strip() if buffer else unit
        if len(candidate) > max_chars and buffer:
            chunks.append(buffer)
            buffer = unit
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)

    return merge_small_chunks(chunks, max_chars, min_chars)


def merge_small_chunks(chunks: List[str], max_chars: int, min_chars: int) -> List[str]:
    """Fold any chunk shorter than min_chars into a neighbour, not just a trailing one.

    Greedy packing can strand a short unit (e.g. a section heading) as its own
    chunk when neither the previous nor the next unit alone leaves room for it.
    Prefer merging with the previous chunk (keeps reading order intact); fall
    back to the next chunk if the previous one is already too full.
    """
    merged: List[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if len(chunk) < min_chars:
            if merged and len(merged[-1]) + 2 + len(chunk) <= max_chars:
                merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
                i += 1
                continue
            if i + 1 < len(chunks) and len(chunk) + 2 + len(chunks[i + 1]) <= max_chars:
                merged.append(f"{chunk}\n\n{chunks[i + 1]}".strip())
                i += 2
                continue
        merged.append(chunk)
        i += 1
    return merged


class DocumentChunker:
    """Reads a document-level CSV and writes a chunk-level CSV from it."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def _load_rows(self) -> tuple[List[Dict[str, str]], List[str]]:
        with open(self.config.input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows, list(reader.fieldnames or [])

    def _resolve_text_column(self, fieldnames: List[str]) -> str:
        if self.config.text_column:
            if self.config.text_column not in fieldnames:
                raise ValueError(f"Text column {self.config.text_column!r} not found; available columns: {fieldnames}")
            return self.config.text_column
        for candidate in AUTO_DETECT_TEXT_COLUMNS:
            if candidate in fieldnames:
                return candidate
        raise ValueError(
            f"Could not auto-detect a text column (looked for {AUTO_DETECT_TEXT_COLUMNS}); "
            f"pass --text-column explicitly. Available columns: {fieldnames}"
        )

    def _resolve_id_column(self, fieldnames: List[str]) -> str:
        if self.config.id_column not in fieldnames:
            raise ValueError(f"Id column {self.config.id_column!r} not found; available columns: {fieldnames}")
        return self.config.id_column

    def chunk_rows(self) -> List[Dict[str, Any]]:
        rows, fieldnames = self._load_rows()
        text_column = self._resolve_text_column(fieldnames)
        id_column = self._resolve_id_column(fieldnames)
        other_columns = [c for c in fieldnames if c != text_column]

        output_rows: List[Dict[str, Any]] = []
        seen_chunk_ids: set = set()
        for row in rows:
            doc_id = row[id_column]
            chunks = chunk_text(row.get(text_column, ""), self.config.max_chunk_chars, self.config.min_chunk_chars)
            total_chunks = len(chunks)
            for chunk_index, chunk in enumerate(chunks):
                # doc_id is expected to be unique, but source data occasionally repeats one
                # (e.g. two published cases sharing a slug); disambiguate rather than collide.
                base_chunk_id = f"{doc_id}_chunk{chunk_index:03d}"
                chunk_id = base_chunk_id
                dup_suffix = 1
                while chunk_id in seen_chunk_ids:
                    chunk_id = f"{base_chunk_id}_dup{dup_suffix}"
                    dup_suffix += 1
                seen_chunk_ids.add(chunk_id)

                out_row: Dict[str, Any] = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "chunk_text": chunk,
                }
                for col in other_columns:
                    out_row[col] = row[col]
                output_rows.append(out_row)
        return output_rows

    @staticmethod
    def save_to_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
        fieldnames = list(rows[0].keys()) if rows else ["chunk_id", "doc_id", "chunk_index", "total_chunks", "chunk_text"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run(self) -> List[Dict[str, Any]]:
        """Chunk and write CSV in one call. Returns the chunk rows too."""
        rows = self.chunk_rows()
        output_path = self.config.output_csv or self._default_output_path()
        self.save_to_csv(rows, output_path)
        return rows

    def _default_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_chunks{ext or '.csv'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = ChunkingConfig(input_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Path to the document-level CSV to chunk")
    parser.add_argument("--text-column", default=defaults.text_column, help="Column holding the case text (auto-detected as 'txt' or 'description' if omitted)")
    parser.add_argument("--id-column", default=defaults.id_column, help="Column holding each document's unique id")
    parser.add_argument("--output", default=defaults.output_csv, help="Output CSV path (default: '<input>_chunks.csv')")
    parser.add_argument("--max-chunk-chars", type=int, default=defaults.max_chunk_chars, help="Target maximum characters per chunk")
    parser.add_argument("--min-chunk-chars", type=int, default=defaults.min_chunk_chars, help="Trailing chunks shorter than this are merged into the previous chunk")
    args = parser.parse_args()

    config = ChunkingConfig(
        input_csv=args.input_csv,
        text_column=args.text_column,
        id_column=args.id_column,
        output_csv=args.output,
        max_chunk_chars=args.max_chunk_chars,
        min_chunk_chars=args.min_chunk_chars,
    )
    chunker = DocumentChunker(config)
    rows = chunker.run()
    output_path = config.output_csv or chunker._default_output_path()
    num_docs = len({r["doc_id"] for r in rows})
    print(f"Chunked {num_docs} documents into {len(rows)} chunks -> {output_path}")


if __name__ == "__main__":
    main()
