"""Generate one few-shot query per chunk using a local Ollama LLM.

Reads a chunk-level CSV produced by chunk_documents.py (e.g.
asylex_raw_documents_sample_chunks.csv, fln_praksis_2026_chunks.csv, or
euaa_asylum_report_chunks.csv -- any chunk-level CSV with a `chunk_text`
column works), builds a few-shot prompt for each chunk's `chunk_text` from
fewshot_qgen.txt (the instruction template) and fewshot_examples.txt (the
few-shot passage/query pairs), sends it to a local Ollama model, and writes a
new CSV with one generated query per chunk alongside every original column.

The Ollama model and the input CSV are both arguments, so the same script
works against any chunked dataset with any locally-pulled model.

Usage as a library:

    from generate_queries import QueryGenConfig, QueryGenerator

    config = QueryGenConfig(input_csv="asylex_raw_documents_sample_chunks.csv", model="gemma3:4b")
    rows = QueryGenerator(config).run()   # also writes the output CSV

Usage from the command line:

    python3 generate_queries.py asylex_raw_documents_sample_chunks.csv
    python3 generate_queries.py fln_praksis_2026_chunks.csv --model llama3.1:8b --limit 20

Requires a running Ollama server (`ollama serve`) with the chosen model
pulled (`ollama pull gemma3:4b`).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class QueryGenConfig:
    """What to generate queries for, and with which model/prompt."""

    input_csv: str
    model: str = "gemma3:4b"
    chunk_text_column: str = "chunk_text"
    prompt_template_path: str = "prompts/fewshot_qgen.txt"
    examples_path: str = "prompts/fewshot_examples.txt"
    ollama_host: str = "http://localhost:11434"
    output_csv: Optional[str] = None  # default: "<input, minus _chunks>_queries.csv"
    limit: Optional[int] = None  # only process the first N chunks (handy for testing)
    max_workers: int = 4  # concurrent requests to the Ollama server
    temperature: float = 0.2
    request_timeout: int = 120


def parse_examples(examples_text: str) -> List[Tuple[str, str]]:
    """Parse "Passage N: ...\\nQuery N: ..." blocks (see fewshot_examples.txt) into pairs."""
    passages = re.findall(r"Passage \d+:\s*(.+?)(?=\nQuery \d+:)", examples_text, re.S)
    queries = re.findall(r"Query \d+:\s*(.+?)(?:\n\n|\Z)", examples_text, re.S)
    return [(p.strip(), q.strip()) for p, q in zip(passages, queries)]


def build_prompt(template: str, chunk_text: str, examples: List[Tuple[str, str]]) -> str:
    """Fill fewshot_qgen.txt's target {passage} and repeat its example line per few-shot pair."""
    marker = "Examples:\n"
    marker_end = template.index(marker) + len(marker)
    head, tail = template[:marker_end], template[marker_end:]
    example_line, _, rest = tail.partition("\n")

    examples_block = "\n".join(example_line.format(passage=p, query=q) for p, q in examples)
    return f"{head.format(passage=chunk_text)}{examples_block}{rest}"


def call_ollama(model: str, prompt: str, host: str, temperature: float, timeout: int) -> str:
    response = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


class QueryGenerator:
    """Loads chunks + the few-shot prompt, queries Ollama per chunk, and writes the result CSV."""

    def __init__(self, config: QueryGenConfig):
        self.config = config

    def _load_prompt_and_examples(self) -> Tuple[str, List[Tuple[str, str]]]:
        template = Path(self.config.prompt_template_path).read_text(encoding="utf-8")
        examples = parse_examples(Path(self.config.examples_path).read_text(encoding="utf-8"))
        if not examples:
            raise ValueError(f"No few-shot examples parsed from {self.config.examples_path}")
        return template, examples

    def _load_rows(self) -> List[Dict[str, str]]:
        with open(self.config.input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if self.config.chunk_text_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Column {self.config.chunk_text_column!r} not found; available columns: {reader.fieldnames}"
                )
            rows = list(reader)
        return rows[: self.config.limit] if self.config.limit is not None else rows

    def generate(self) -> List[Dict[str, Any]]:
        template, examples = self._load_prompt_and_examples()
        rows = self._load_rows()
        cfg = self.config

        def process(row: Dict[str, str]) -> Dict[str, Any]:
            prompt = build_prompt(template, row[cfg.chunk_text_column], examples)
            try:
                query = call_ollama(cfg.model, prompt, cfg.ollama_host, cfg.temperature, cfg.request_timeout)
            except Exception as exc:  # network error, timeout, model not pulled, etc.
                print(f"warning: query generation failed for {row.get('chunk_id')}: {exc}", file=sys.stderr)
                query = ""
            chunk_id = row.get("chunk_id", "")
            out: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_id": row.get("doc_id", ""),
                "query_id": f"{chunk_id}_query",
                "generated_query": query,
            }
            for col, val in row.items():
                if col not in out:
                    out[col] = val
            return out

        results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {executor.submit(process, row): i for i, row in enumerate(rows)}
            done = 0
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if done % 25 == 0 or done == len(rows):
                    print(f"  generated {done}/{len(rows)} queries", file=sys.stderr)

        return [r for r in results if r is not None]

    @staticmethod
    def save_to_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
        fieldnames = list(rows[0].keys()) if rows else ["chunk_id", "doc_id", "query_id", "generated_query"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run(self) -> List[Dict[str, Any]]:
        """Generate and write CSV in one call. Returns the rows too."""
        rows = self.generate()
        self.save_to_csv(rows, self.config.output_csv or self._default_output_path())
        return rows

    def _default_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        stem = re.sub(r"_chunks$", "", stem)
        return f"{stem}_queries{ext or '.csv'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = QueryGenConfig(input_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Chunk-level CSV to generate queries for (any CSV with a chunk_text column)")
    parser.add_argument("--model", default=defaults.model, help="Ollama model to use, e.g. gemma3:4b")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<input, minus _chunks>_queries.csv')")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chunks (useful for testing)")
    parser.add_argument("--max-workers", type=int, default=defaults.max_workers, help="Concurrent requests to the Ollama server")
    parser.add_argument("--ollama-host", default=defaults.ollama_host, help="Ollama server base URL")
    args = parser.parse_args()

    config = QueryGenConfig(
        input_csv=args.input_csv,
        model=args.model,
        output_csv=args.output,
        limit=args.limit,
        max_workers=args.max_workers,
        ollama_host=args.ollama_host,
    )
    generator = QueryGenerator(config)
    rows = generator.run()
    output_path = config.output_csv or generator._default_output_path()
    print(f"Generated {len(rows)} queries -> {output_path}")


if __name__ == "__main__":
    main()
