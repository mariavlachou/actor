"""Dense indexing and retrieval with PyTerrier: E5 or Qwen3, via FlexIndex.

Indexes a *_chunks_docs.csv file (columns: docno, text -- see chunks_to_docs.py)
into a PyTerrier `pyterrier_dr.FlexIndex` dense vector index, retrieves the
top-k results per query from a matching *_queries_topic_names_qid_query.csv
file (columns: qid, query -- see topic_to_queries.py), and returns/saves a
DataFrame of the results with the retrieved document's text joined back in.

Two encoder options:
  - "e5":    pyterrier_dr.E5, PyTerrier's built-in E5 bi-encoder
             (intfloat/e5-{small,base,large}-v2; default "base").
  - "qwen3": a Qwen3 embedding checkpoint from Hugging Face, loaded through
             pyterrier_dr's generic sentence-transformers wrapper
             (default: Qwen/Qwen3-Embedding-0.6B), since pyterrier_dr has no
             dedicated Qwen3 class.

Both encoders use FlexIndex's standard encode-then-index / encode-then-search
pipeline shape: `encoder.doc_encoder() >> index.indexer(...)` for indexing,
`encoder.query_encoder() >> index.np_retriever(...)` for retrieval.

Usage as a library:

    from pt_dense_retrieval import DenseRetrievalConfig, DenseRetrievalPipeline

    config = DenseRetrievalConfig(
        docs_csv="euaa_asylum_report_chunks_docs.csv",
        topics_csv="euaa_asylum_report_queries_topic_names_qid_query.csv",
        retriever="e5",
    )
    df = DenseRetrievalPipeline(config).run()   # also writes the output CSV

Usage from the command line:

    python3 pt_dense_retrieval.py --docs euaa_asylum_report_chunks_docs.csv \
        --topics euaa_asylum_report_queries_topic_names_qid_query.csv --retriever e5

    python3 pt_dense_retrieval.py --docs fln_praksis_2026_chunks_docs.csv \
        --topics fln_praksis_2026_queries_topic_names_qid_query.csv \
        --retriever qwen3 --qwen3-model Qwen/Qwen3-Embedding-0.6B

Requires: python-terrier, pyterrier_dr, sentence-transformers, pandas, and a
working Java installation (FlexIndex itself is JVM-free, but importing
`pyterrier` still starts one).
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyterrier as pt
from pyterrier_dr import E5, FlexIndex, SBertBiEncoder


@dataclass
class DenseRetrievalConfig:
    """What to index, which dense encoder to use, and how big a result list to return."""

    docs_csv: str  # a *_chunks_docs.csv file: columns docno, text
    topics_csv: str  # a *_queries_topic_names_qid_query.csv file: columns qid, query

    retriever: str = "e5"  # "e5" | "qwen3"
    top_k: int = 100

    e5_model: str = "base"  # "small" | "base" | "large" (an E5.VARIANTS key), or any HF checkpoint id
    qwen3_model: str = "Qwen/Qwen3-Embedding-0.6B"

    index_dir: Optional[str] = None  # default: "./pt_dense_index_<docs stem>_<retriever>"
    overwrite_index: bool = False

    batch_size: int = 32
    output_csv: Optional[str] = None  # default: "<topics stem>_<retriever>.csv"


class DenseRetrievalPipeline:
    """Builds a FlexIndex dense index (E5 or Qwen3) and retrieves the top-k per query."""

    def __init__(self, config: DenseRetrievalConfig):
        self.config = config
        if not pt.started():
            pt.init()

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
        df["query"] = df["query"].fillna("").astype(str)
        return df[df["query"] != ""].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def _get_encoder(self):
        cfg = self.config
        if cfg.retriever == "e5":
            model_name = E5.VARIANTS.get(cfg.e5_model, cfg.e5_model)
            return E5(model_name, batch_size=cfg.batch_size)
        elif cfg.retriever == "qwen3":
            return SBertBiEncoder(cfg.qwen3_model, batch_size=cfg.batch_size)
        raise ValueError(f"Unknown retriever {cfg.retriever!r}; expected 'e5' or 'qwen3'")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _default_index_dir(self) -> str:
        stem = os.path.splitext(os.path.basename(self.config.docs_csv))[0]
        return os.path.abspath(f"./index_dir/pt_dense_index_{stem}_{self.config.retriever}")

    def build_index(self, docs: pd.DataFrame, encoder) -> str:
        cfg = self.config
        index_dir = os.path.abspath(cfg.index_dir or self._default_index_dir())

        if cfg.overwrite_index and os.path.isdir(index_dir):
            shutil.rmtree(index_dir)
        if os.path.isdir(index_dir) and os.listdir(index_dir):
            return index_dir  # reuse an existing index for this (docs, retriever) pair

        os.makedirs(index_dir, exist_ok=True)
        index = FlexIndex(index_dir)
        indexing_pipeline = encoder.doc_encoder(batch_size=cfg.batch_size) >> index.indexer(mode="overwrite")
        indexing_pipeline.index(docs.to_dict("records"))
        return index_dir

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def retrieve(self) -> pd.DataFrame:
        cfg = self.config
        docs = self._load_docs()
        topics = self._load_topics()

        encoder = self._get_encoder()
        index_dir = self.build_index(docs, encoder)
        index = FlexIndex(index_dir)

        retrieval_pipeline = encoder.query_encoder(batch_size=cfg.batch_size) >> index.np_retriever(num_results=cfg.top_k)
        results = retrieval_pipeline.transform(topics)  # already carries the qid's query text through
        results = results.drop(columns=["query_vec", "doc_vec"], errors="ignore")  # raw embeddings, not needed in the output
        results = results.merge(docs[["docno", "text"]], on="docno", how="left")

        return results.reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """Retrieve and write CSV in one call. Returns the DataFrame too."""
        df = self.retrieve()
        df.to_csv(self.config.output_csv or self._default_output_path(), index=False)
        return df

    def _default_output_path(self) -> str:
        stem = os.path.splitext(self.config.topics_csv)[0]
        return f"{stem}_{self.config.retriever}.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    defaults = DenseRetrievalConfig(docs_csv="", topics_csv="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True, help="*_chunks_docs.csv file to index (columns: docno, text)")
    parser.add_argument("--topics", required=True, help="*_queries_topic_names_qid_query.csv query set (columns: qid, query)")
    parser.add_argument("--retriever", choices=["e5", "qwen3"], default=defaults.retriever, help="Dense retrieval model")
    parser.add_argument("--top-k", type=int, default=defaults.top_k, help="Number of results to return per query")
    parser.add_argument("--e5-model", default=defaults.e5_model, help="E5 variant ('small'/'base'/'large') or a full HF checkpoint id")
    parser.add_argument("--qwen3-model", default=defaults.qwen3_model, help="Qwen3 embedding checkpoint on Hugging Face")
    parser.add_argument("--index-dir", default=None, help="Where to build/reuse the FlexIndex (default: ./pt_dense_index_<docs stem>_<retriever>)")
    parser.add_argument("--overwrite-index", action="store_true", help="Rebuild the index even if one already exists at --index-dir")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Encoding batch size")
    parser.add_argument("--output", default=None, help="Output CSV path (default: '<topics stem>_<retriever>.csv')")
    args = parser.parse_args()

    config = DenseRetrievalConfig(
        docs_csv=args.docs,
        topics_csv=args.topics,
        retriever=args.retriever,
        top_k=args.top_k,
        e5_model=args.e5_model,
        qwen3_model=args.qwen3_model,
        index_dir=args.index_dir,
        overwrite_index=args.overwrite_index,
        batch_size=args.batch_size,
        output_csv=args.output,
    )
    pipeline = DenseRetrievalPipeline(config)
    df = pipeline.run()
    output_path = config.output_csv or pipeline._default_output_path()
    print(f"Retrieved {len(df)} rows ({df['qid'].nunique()} queries) -> {output_path}")


if __name__ == "__main__":
    main()
