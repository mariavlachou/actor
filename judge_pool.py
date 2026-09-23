"""LLM-as-a-judge relevance labelling for a *_pool.csv file, using a local Ollama LLM.

Reads a pool CSV produced by pool_results.py (columns: qid, query, docno,
text, rank, score), asks a local Ollama model to judge the relevance of each
row's `text` to its `query` using the prompt in qrels_prompt.txt (a 0-5
relevance scale returned as a JSON object), stores that score in a new
`label` column, then selects just `qid, docno, label` and saves that as a
new CSV.

Usage as a library:

    from judge_pool import JudgeConfig, QrelsJudge

    config = JudgeConfig(input_csv="euaa_asylum_report_queries_topic_names_qid_query_pool.csv")
    labels_df = QrelsJudge(config).run()   # also writes the output CSV

Usage from the command line:

    python3 judge_pool.py euaa_asylum_report_queries_topic_names_qid_query_pool.csv
    python3 judge_pool.py fln_praksis_2026_queries_topic_names_qid_query_sample30_pool.csv \
        --model llama3.1:8b --limit 20

Requires a running Ollama server (`ollama serve`) with the chosen model
pulled (`ollama pull llama3.1:8b`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from generate_queries import call_ollama

# qrels_prompt.txt embeds its two fill-ins as literal `row['user query']` /
# `row['system answer']` text (with either straight or curly apostrophes).
QUERY_PLACEHOLDER_RE = re.compile(r"row\[[’']user query[’']\]")
ANSWER_PLACEHOLDER_RE = re.compile(r"row\[[’']system answer[’']\]")

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)
STANDALONE_SCORE_RE = re.compile(r"\b([0-5])\b")


@dataclass
class JudgeConfig:
    """Which pool to judge, with which model/prompt, and how many rows in parallel."""

    input_csv: str  # a *_pool.csv file: columns qid, query, docno, text, rank, score
    model: str = "llama3.1:8b"
    prompt_template_path: str = "qrels_prompt.txt"
    ollama_host: str = "http://localhost:11434"
    output_csv: Optional[str] = None  # default: "<input>_labels.csv"
    limit: Optional[int] = None  # only judge the first N rows (handy for testing)
    max_workers: int = 4
    temperature: float = 0.0
    request_timeout: int = 120


def build_judge_prompt(template: str, query: str, text: str) -> str:
    prompt = QUERY_PLACEHOLDER_RE.sub(lambda _m: query, template)
    prompt = ANSWER_PLACEHOLDER_RE.sub(lambda _m: text, prompt)
    return prompt


def parse_judge_response(response_text: str) -> Optional[int]:
    """Pull the 0-5 "score" out of the model's (nominally JSON) response."""
    match = JSON_OBJECT_RE.search(response_text)
    if match:
        try:
            data = json.loads(match.group(0))
            score = data.get("score")
            if score is not None:
                return int(score)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback for a model that answered with just a bare digit.
    match = STANDALONE_SCORE_RE.search(response_text)
    return int(match.group(1)) if match else None


class QrelsJudge:
    """Loads a pool CSV, judges each (query, text) pair with Ollama, and writes qid/docno/label."""

    def __init__(self, config: JudgeConfig):
        self.config = config

    def _load_prompt_template(self) -> str:
        return Path(self.config.prompt_template_path).read_text(encoding="utf-8")

    def _load_pool(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.input_csv, dtype={"qid": str, "docno": str})
        missing = [c for c in ("qid", "query", "docno", "text") if c not in df.columns]
        if missing:
            raise ValueError(f"input_csv missing column(s) {missing}; available columns: {list(df.columns)}")
        return df.iloc[: self.config.limit].reset_index(drop=True) if self.config.limit is not None else df

    def label(self) -> pd.DataFrame:
        cfg = self.config
        template = self._load_prompt_template()
        df = self._load_pool()

        def judge_row(row: pd.Series) -> Optional[int]:
            query = str(row["query"])
            text = "" if pd.isna(row["text"]) else str(row["text"])
            prompt = build_judge_prompt(template, query, text)
            try:
                response = call_ollama(cfg.model, prompt, cfg.ollama_host, cfg.temperature, cfg.request_timeout)
            except Exception as exc:  # network error, timeout, model not pulled, etc.
                print(f"warning: judging failed for qid={row['qid']} docno={row['docno']}: {exc}", file=sys.stderr)
                return None
            score = parse_judge_response(response)
            if score is None:
                print(f"warning: could not parse a score for qid={row['qid']} docno={row['docno']}: {response[:200]!r}", file=sys.stderr)
            return score

        labels: List[Optional[int]] = [None] * len(df)
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {executor.submit(judge_row, row): i for i, row in df.iterrows()}
            done = 0
            for future in as_completed(futures):
                labels[futures[future]] = future.result()
                done += 1
                if done % 25 == 0 or done == len(df):
                    print(f"  judged {done}/{len(df)} rows", file=sys.stderr)

        df = df.copy()
        df["label"] = labels
        return df

    def run(self) -> pd.DataFrame:
        """Judge, select qid/docno/label, and write CSV in one call. Returns that DataFrame too."""
        labeled = self.label()
        output_df = labeled[["qid", "docno", "label"]]
        output_df.to_csv(self.config.output_csv or self._default_output_path(), index=False)
        return output_df

    def _default_output_path(self) -> str:
        stem, ext = os.path.splitext(self.config.input_csv)
        return f"{stem}_labels{ext or '.csv'}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = JudgeConfig(input_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Pool CSV to judge (columns: qid, query, docno, text, rank, score)")
    parser.add_argument("--model", default=defaults.model, help="Ollama model to use, e.g. llama3.1:8b")
    parser.add_argument("--prompt-path", default=defaults.prompt_template_path, help="Path to the judge prompt template")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<input>_labels.csv')")
    parser.add_argument("--limit", type=int, default=None, help="Only judge the first N rows (useful for testing)")
    parser.add_argument("--max-workers", type=int, default=defaults.max_workers, help="Concurrent requests to the Ollama server")
    parser.add_argument("--ollama-host", default=defaults.ollama_host, help="Ollama server base URL")
    args = parser.parse_args()

    config = JudgeConfig(
        input_csv=args.input_csv,
        model=args.model,
        prompt_template_path=args.prompt_path,
        output_csv=args.output,
        limit=args.limit,
        max_workers=args.max_workers,
        ollama_host=args.ollama_host,
    )
    judge = QrelsJudge(config)
    df = judge.run()
    output_path = config.output_csv or judge._default_output_path()
    print(f"Labelled {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
