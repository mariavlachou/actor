"""Sparse indexing and retrieval with PyTerrier: BM25 or SPLADE, with optional monoT5 reranking.

Indexes a *_chunks_docs.csv file (columns: docno, text -- see chunks_to_docs.py)
with PyTerrier's Terrier backend, retrieves the top-k results per query from a
matching *_queries_topic_names_qid_query.csv file (columns: qid, query -- see
topic_to_queries.py), and returns/saves a DataFrame of the results with the
retrieved document's text joined back in.

Language handling: if the collection is Danish (the fln.dk case), both BM25
and SPLADE are indexed with Terrier's DanishSnowballStemmer; otherwise the
default sparse-indexing stemmer (Porter, English) is used. Danish detection
is automatic from the docs filename ("fln_praksis"), or can be forced with
--language.

Note on SPLADE + the Danish stemmer: SPLADE indexes pre-tokenised neural
subword weights (via `pretokenised=True`), which bypasses Terrier's term
pipeline entirely, so a linguistic stemmer has no real effect there. The
DanishSnowballStemmer is still passed through to the SPLADE indexer as
instructed (for consistency across both retrievers), but in practice only
the BM25 index is meaningfully affected by it.

monoT5 reranking (--rerank-monot5) re-scores the already-retrieved top-k
(the result set size doesn't change) using pyterrier_t5.MonoT5ReRanker: the
Danish case explicitly loads castorini/monot5-base-msmarco; the non-Danish
case uses pyterrier_t5's own default model (which currently happens to also
be castorini/monot5-base-msmarco -- the two branches are kept structurally
distinct per the given instructions even though they resolve to the same
checkpoint today).

Usage as a library:

    from pt_retrieval import RetrievalConfig, PyTerrierRetrievalPipeline

    config = RetrievalConfig(
        docs_csv="fln_praksis_2026_chunks_docs.csv",
        topics_csv="fln_praksis_2026_queries_topic_names_qid_query.csv",
        retriever="bm25",
        rerank_monot5=True,
    )
    df = PyTerrierRetrievalPipeline(config).run()   # also writes the output CSV

Usage from the command line:

    python3 pt_retrieval.py --docs euaa_asylum_report_chunks_docs.csv \
        --topics euaa_asylum_report_queries_topic_names_qid_query.csv --retriever bm25

    python3 pt_retrieval.py --docs fln_praksis_2026_chunks_docs.csv \
        --topics fln_praksis_2026_queries_topic_names_qid_query.csv \
        --retriever splade --rerank-monot5

Requires: python-terrier, pyterrier_t5, pyterrier-splade, pandas, a working
Java installation (PyTerrier's Terrier backend runs on the JVM).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyterrier as pt

DANISH_FILENAME_MARKER = "fln_praksis"  # the fln.dk (Flygtningenaevnet) collection is the Danish one here

# Terrier's query parser treats punctuation like &, ', (, ) as query syntax;
# auto-generated topic names can contain it, so queries are sanitised to plain
# alphanumerics before retrieval.
_QUERY_SANITIZE_RE = re.compile(r"[^\w\s-]", re.UNICODE)


@dataclass
class RetrievalConfig:
    """What to index, what to search it with, and how to rank the results."""

    docs_csv: str  # a *_chunks_docs.csv file: columns docno, text
    topics_csv: str  # a *_queries_topic_names_qid_query.csv file: columns qid, query

    retriever: str = "bm25"  # "bm25" | "splade"
    top_k: int = 100

    language: str = "auto"  # "auto" | "danish" | "other"

    index_dir: Optional[str] = None  # default: "./pt_index_<docs stem>_<retriever>"
    overwrite_index: bool = False

    rerank_monot5: bool = False
    splade_model: str = "naver/splade-cocondenser-ensembledistil"

    output_csv: Optional[str] = None  # default: "<topics stem>_<retriever>[_monot5].csv"


def is_danish_collection(docs_csv: str, language: str) -> bool:
    if language == "danish":
        return True
    if language == "other":
        return False
    return DANISH_FILENAME_MARKER in os.path.basename(docs_csv).lower()


def sanitize_query(query: str) -> str:
    return _QUERY_SANITIZE_RE.sub(" ", str(query)).strip()


class PyTerrierRetrievalPipeline:
    """Builds a Terrier index (BM25 or SPLADE), retrieves top-k, and optionally reranks with monoT5."""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        if not pt.started():
            pt.init()
        self._splade_factory = None  # lazily created; shared between indexing and retrieval

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_docs(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.docs_csv, dtype={"docno": str})
        missing = [c for c in ("docno", "text") if c not in df.columns]
        if missing:
            raise ValueError(f"docs_csv missing column(s) {missing}; available columns: {list(df.columns)}")
        df["text"] = df["text"].fillna("")
        return df

    def _load_topics(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.topics_csv, dtype={"qid": str})
        missing = [c for c in ("qid", "query") if c not in df.columns]
        if missing:
            raise ValueError(f"topics_csv missing column(s) {missing}; available columns: {list(df.columns)}")
        df["query"] = df["query"].fillna("").apply(sanitize_query)
        return df[df["query"] != ""].reset_index(drop=True)

    def _is_danish(self) -> bool:
        return is_danish_collection(self.config.docs_csv, self.config.language)

    # ------------------------------------------------------------------
    # SPLADE model (shared between indexing and retrieval)
    # ------------------------------------------------------------------
    def _get_splade_factory(self):
        if self._splade_factory is None:
            from pyterrier_splade import SpladeFactory

            self._splade_factory = SpladeFactory(model=self.config.splade_model)
        return self._splade_factory

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _default_index_dir(self) -> str:
        stem = os.path.splitext(os.path.basename(self.config.docs_csv))[0]
        return os.path.abspath(f"./index_dir/pt_index_{stem}_{self.config.retriever}")

    def build_index(self, docs: pd.DataFrame) -> str:
        cfg = self.config
        index_dir = os.path.abspath(cfg.index_dir or self._default_index_dir())

        if cfg.overwrite_index and os.path.isdir(index_dir):
            shutil.rmtree(index_dir)
        if os.path.isdir(index_dir) and os.listdir(index_dir):
            return index_dir  # reuse an existing index for this (docs, retriever) pair

        os.makedirs(index_dir, exist_ok=True)
        stemmer = pt.TerrierStemmer.danish if self._is_danish() else pt.TerrierStemmer.porter

        if cfg.retriever == "bm25":
            indexer = pt.IterDictIndexer(index_dir, meta={"docno": 64}, stemmer=stemmer, overwrite=True)
            indexer.index(docs.to_dict("records"))
        elif cfg.retriever == "splade":
            base_indexer = pt.IterDictIndexer(
                index_dir, meta={"docno": 64}, pretokenised=True, overwrite=True, stemmer=stemmer
            )
            indexing_pipe = self._get_splade_factory().indexing(text_field="text") >> base_indexer
            indexing_pipe.index(docs.to_dict("records"))
        else:
            raise ValueError(f"Unknown retriever {cfg.retriever!r}; expected 'bm25' or 'splade'")

        return index_dir

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _get_retriever_pipeline(self, index_dir: str):
        cfg = self.config
        if cfg.retriever == "bm25":
            return pt.terrier.Retriever(index_dir, wmodel="BM25", num_results=cfg.top_k)
        elif cfg.retriever == "splade":
            # termpipelines="" disables query-side stemming/stopwording: SPLADE's query encoder
            # already produces the final sparse subword-weight representation to match.
            retr = pt.terrier.Retriever(
                index_dir, wmodel="Tf", num_results=cfg.top_k, properties={"termpipelines": ""}
            )
            return self._get_splade_factory().query() >> retr
        raise ValueError(f"Unknown retriever {cfg.retriever!r}; expected 'bm25' or 'splade'")

    def _get_monot5(self):
        from pyterrier_t5 import MonoT5ReRanker

        if self._is_danish():
            return MonoT5ReRanker(model="castorini/monot5-base-msmarco")
        return MonoT5ReRanker()  # pyterrier_t5's own built-in default checkpoint

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def retrieve(self) -> pd.DataFrame:
        cfg = self.config
        docs = self._load_docs()
        topics = self._load_topics()

        index_dir = self.build_index(docs)
        retrieval_pipeline = self._get_retriever_pipeline(index_dir) % cfg.top_k

        results = retrieval_pipeline.transform(topics)  # already carries the qid's query text through
        results = results.merge(docs[["docno", "text"]], on="docno", how="left")

        if cfg.rerank_monot5:
            monot5 = self._get_monot5()
            results = monot5.transform(results)
            results = results.sort_values(["qid", "rank"] if "rank" in results.columns else ["qid", "score"], ascending=[True, True] if "rank" in results.columns else [True, False])

        return results.reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """Retrieve (and optionally rerank) in one call, writing the output CSV. Returns the DataFrame too."""
        df = self.retrieve()
        df.to_csv(self.config.output_csv or self._default_output_path(), index=False)
        return df

    def _default_output_path(self) -> str:
        stem = os.path.splitext(self.config.topics_csv)[0]
        suffix = f"_{self.config.retriever}" + ("_monot5" if self.config.rerank_monot5 else "")
        return f"{stem}{suffix}.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = RetrievalConfig(docs_csv="", topics_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True, help="*_chunks_docs.csv file to index (columns: docno, text)")
    parser.add_argument("--topics", required=True, help="*_queries_topic_names_qid_query.csv query set (columns: qid, query)")
    parser.add_argument("--retriever", choices=["bm25", "splade"], default=defaults.retriever, help="Sparse retrieval model")
    parser.add_argument("--top-k", type=int, default=defaults.top_k, help="Number of results to return per query")
    parser.add_argument("--language", choices=["auto", "danish", "other"], default=defaults.language, help="Force the indexing language, or auto-detect from --docs filename")
    parser.add_argument("--index-dir", default=None, help="Where to build/reuse the Terrier index (default: ./pt_index_<docs stem>_<retriever>)")
    parser.add_argument("--overwrite-index", action="store_true", help="Rebuild the index even if one already exists at --index-dir")
    parser.add_argument("--rerank-monot5", action="store_true", help="Rerank the top-k results with monoT5")
    parser.add_argument("--splade-model", default=defaults.splade_model, help="SPLADE model checkpoint")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<topics stem>_<retriever>[_monot5].csv')")
    args = parser.parse_args()

    config = RetrievalConfig(
        docs_csv=args.docs,
        topics_csv=args.topics,
        retriever=args.retriever,
        top_k=args.top_k,
        language=args.language,
        index_dir=args.index_dir,
        overwrite_index=args.overwrite_index,
        rerank_monot5=args.rerank_monot5,
        splade_model=args.splade_model,
        output_csv=args.output,
    )
    pipeline = PyTerrierRetrievalPipeline(config)
    df = pipeline.run()
    output_path = config.output_csv or pipeline._default_output_path()
    print(f"Retrieved {len(df)} rows ({df['qid'].nunique()} queries) -> {output_path}")


if __name__ == "__main__":
    main()
