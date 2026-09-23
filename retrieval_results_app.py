"""Streamlit app: visualise retrieval results for a query, and visualise
which topics a given case file answers.

Auto-discovers every dataset that has a judged pool (a "*_pool_labels.csv"
file, produced by judge_pool.py) sitting in the same directory as this
script file (not the working directory it's launched from), since a
dataset can't be meaningfully visualised without relevance labels. Both
tabs share chunk_id <-> doc_id <-> case metadata mapping via the dataset's
*_chunks.csv file (the one place a retrieved chunk and its parent "case
file" -- the original scraped/sampled document -- are both present).

Tab 1, "Topics -> Case Files": pick a topic and a retrieval method; the top 3
distinct case files by their single most relevant (highest-labelled) chunk
for that topic/method pair.

Tab 2, "Case File -> Topics": pick a case file (any document with at least
one labelled chunk); see its full reconstructed text, with whichever chunk(s)
reached that case's own highest judge label highlighted inline (not a fixed
top-N -- some cases only have a couple of chunks total), and which topic(s)
each of those chunks was judged highly relevant to.

These are two different navigation directions (query-first vs.
document-first) over the same underlying qid/docno/label data, so they live
in separate tabs rather than one crowded page -- showing both sets of
dropdowns at once would leave it unclear which selection drives which
section. The dataset selector is shared above both tabs, since switching
datasets should reset both views together.

Run with (from any directory -- data paths resolve relative to this file,
not the working directory streamlit was launched from):

    streamlit run retrieval_results_app.py
    streamlit run /path/to/actor/retrieval_results_app.py
"""

from __future__ import annotations

import glob
import html
import os
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from generate_queries import call_ollama

# Resolve data files relative to this script's own location, not the working
# directory streamlit happened to be launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find(pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(BASE_DIR, pattern)))


# Maps the stem prefix every topics/results/labels file for a base dataset
# shares, to that dataset's chunk-level CSV (chunk_id, doc_id, chunk_text,
# and whatever case metadata that scrape/sample produced).
BASE_DATASET_CHUNKS = {
    "euaa_asylum_report": "euaa_asylum_report_chunks.csv",
    "fln_praksis_2026": "fln_praksis_2026_chunks.csv",
    "asylex_raw_documents_sample": "asylex_raw_documents_sample_chunks.csv",
}

# Case-metadata columns worth showing, in display order, if a dataset's
# chunks CSV happens to have them (schemas vary: EUAA has court/case_number,
# fln has categories, AsyLex has neither).
CASE_METADATA_COLUMNS = [
    ("title", "Title"),
    ("url", "URL"),
    ("country", "Country"),
    ("court", "Court"),
    ("case_type", "Case type"),
    ("case_number", "Case number"),
    ("published_date", "Published"),
    ("categories", "Categories"),
]

TOP_N = 3  # Tab 1 surfaces the top-3 case files by label (Tab 2 highlights by label tier, uncapped)


def inject_styles() -> None:
    # st.html (not st.markdown(unsafe_allow_html=True)) is the reliable way to inject a raw
    # <style> block: markdown's renderer doesn't treat <style> as opaque CSS, so its contents
    # end up rendered as visible page text instead of being applied as a stylesheet.
    st.html(
        textwrap.dedent(
            """\
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
            <style>
              :root {
                --ink: #1c2131;
                --ink-soft: #565f78;
                --accent: #8a5a12;
                --accent-bg: #fdf1dc;
                --line: #e3e1d8;
              }
              .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }
              h1, h2, h3, h4 { letter-spacing: -0.01em; }
              .stTabs [data-baseweb="tab-list"] { gap: 4px; }
              .stTabs [data-baseweb="tab"] {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.92rem;
                padding: 10px 18px;
              }

              .case-doc-text {
                font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
                font-size: 1.05rem;
                line-height: 1.95;
                color: var(--ink);
                white-space: pre-wrap;
                background: #fffefb;
                border: 1px solid var(--line);
                border-radius: 6px;
                padding: 24px 28px;
                margin-top: 6px;
              }
              /* Per-chunk highlight colour is set inline via --hue (one hue per topic). */
              mark.chunk-highlight {
                background: hsl(var(--hue) 70% 85%);
                box-shadow: inset 0 -2px 0 hsl(var(--hue) 55% 45%);
                border-radius: 3px;
                padding: 0 2px;
                text-decoration: none;
                color: var(--ink);
              }
              .rank-badge {
                display: inline-block;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.7rem;
                font-weight: 600;
                color: #fff;
                background: hsl(var(--hue) 55% 40%);
                border-radius: 3px;
                padding: 0 5px;
                vertical-align: super;
                margin-right: 2px;
              }
              .topic-bubble {
                display: inline-flex;
                align-items: center;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.78rem;
                background: hsl(var(--hue) 75% 95%);
                color: hsl(var(--hue) 65% 28%);
                border: 1px solid hsl(var(--hue) 55% 70%);
                border-radius: 999px;
                padding: 1px 10px;
                margin: 0 3px;
                white-space: nowrap;
              }
              .topic-bubble b { font-family: inherit; }

              .topic-chip {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.82rem;
                background: var(--accent-bg);
                color: var(--accent);
                border: 1px solid #eddcb5;
                border-radius: 999px;
                padding: 4px 12px;
                margin: 3px 6px 3px 0;
              }
              .topic-chip b { color: var(--ink); font-family: inherit; }
              .label-pill {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                font-weight: 600;
                color: #fff;
                background: var(--accent);
                border-radius: 4px;
                padding: 1px 6px;
              }

              .meta-line { font-size: 0.92rem; color: var(--ink-soft); margin: 2px 0; }
              .meta-line b { color: var(--ink); }
            </style>
            """
        )
    )


@st.cache_data
def discover_datasets() -> Dict[str, dict]:
    """Find every judged dataset (has a *_pool_labels.csv) and its result/topic/chunk files."""
    datasets: Dict[str, dict] = {}
    for labels_path in _find("*_pool_labels.csv"):
        stem = os.path.basename(labels_path)[: -len("_pool_labels.csv")]
        topics_path = os.path.join(BASE_DIR, f"{stem}.csv")
        if not os.path.exists(topics_path):
            continue

        chunks_filename = next((v for k, v in BASE_DATASET_CHUNKS.items() if stem.startswith(k)), None)
        chunks_path = os.path.join(BASE_DIR, chunks_filename) if chunks_filename else None
        if chunks_path is None or not os.path.exists(chunks_path):
            continue

        methods: Dict[str, str] = {}
        for path in _find(f"{stem}_*.csv"):
            suffix = os.path.basename(path)[len(stem) + 1 : -len(".csv")]
            if suffix in ("pool", "pool_labels") or suffix.endswith("_metrics"):
                continue
            try:
                header = set(pd.read_csv(path, nrows=0).columns)
            except Exception:
                continue
            if not {"qid", "docno", "rank", "score"}.issubset(header):
                continue  # not an actual retrieval result file (e.g. a metrics CSV)
            methods[suffix] = path
        if not methods:
            continue

        display_name = stem.replace("_queries_topic_names_qid_query", "").replace("_", " ")
        datasets[stem] = {
            "label": display_name,
            "topics_csv": topics_path,
            "chunks_csv": chunks_path,
            "labels_csv": labels_path,
            "methods": methods,
        }
    return datasets


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"qid": str, "docno": str, "doc_id": str, "chunk_id": str})


def derive_doc_id(chunk_id: str) -> str:
    """Fallback if a chunks file lacks its own doc_id: strip the "_chunkNNN" suffix."""
    idx = chunk_id.rfind("_chunk")
    return chunk_id[:idx] if idx != -1 else chunk_id


def render_case_metadata(row: pd.Series) -> None:
    for col, label in CASE_METADATA_COLUMNS:
        if col not in row.index or pd.isna(row[col]) or str(row[col]).strip() == "":
            continue
        if col == "url":
            st.html(f'<div class="meta-line"><b>{label}:</b> <a href="{row[col]}">{row[col]}</a></div>')
        else:
            st.html(f'<div class="meta-line"><b>{label}:</b> {row[col]}</div>')


def case_display_title(doc_id: str, case_metadata: pd.DataFrame) -> str:
    if doc_id in case_metadata.index:
        title = case_metadata.loc[doc_id].get("title")
        if pd.notna(title) and str(title).strip():
            return str(title)
    return doc_id


def dedupe_topics_by_text(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Different qids can end up with the same topic text (e.g. two BERTopic clusters that
    happened to get named identically); keep only the first qid per unique topic text so it
    doesn't show up twice wherever topics are listed."""
    return topics_df.drop_duplicates(subset="query", keep="first")


TRANSLATE_MODEL = "gemma3:4b"
TRANSLATE_PROMPT = (
    "Translate the following short Danish phrase into natural, concise English.\n"
    "Only output the resulting phrase itself -- no quotes, labels, or extra commentary.\n\n"
    "Phrase: {text}"
)
_DANISH_CHAR_RE = re.compile(r"[æøåÆØÅ]")


def looks_danish(text: str) -> bool:
    """Danish-specific letters are a near-certain signal; langdetect covers the rest.
    Trusting an LLM to also self-judge "is this already English" for short phrases
    caused it to reword perfectly fine English topics, so detection is kept separate
    from translation."""
    if _DANISH_CHAR_RE.search(text):
        return True
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text) == "da"
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def translate_topic(text: str) -> str:
    """Danish topic names get translated to English for display; English ones pass through
    unchanged (and skip the Ollama call entirely). Cached per unique topic text."""
    if not looks_danish(text):
        return text
    try:
        result = call_ollama(TRANSLATE_MODEL, TRANSLATE_PROMPT.format(text=text), "http://localhost:11434", 0.0, 60)
        result = result.strip().strip('"').strip("'")
        return result or text
    except Exception:
        return text  # Ollama unreachable or the model isn't pulled -- fall back to the original


def with_english_topics(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'query_en' column: each unique topic text translated (English passes through)."""
    unique_texts = topics_df["query"].unique().tolist()
    with st.spinner(f"Checking {len(unique_texts)} topic name(s) for translation…"):
        with ThreadPoolExecutor(max_workers=4) as executor:
            translations = dict(zip(unique_texts, executor.map(translate_topic, unique_texts)))
    df = topics_df.copy()
    df["query_en"] = df["query"].map(translations)
    return df


def render_query_tab(
    dataset: dict,
    topics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    chunk_to_doc: Dict[str, str],
    case_metadata: pd.DataFrame,
) -> None:
    topic_options_df = dedupe_topics_by_text(topics_df)
    query_lookup = dict(zip(topic_options_df["qid"], topic_options_df["query_en"]))

    col1, col2 = st.columns(2)
    with col1:
        selected_qid = st.selectbox(
            "Topic", options=list(query_lookup.keys()), format_func=lambda qid: f"{qid} — {query_lookup[qid]}", key="query_select"
        )
    with col2:
        method = st.selectbox("Retrieval method", options=list(dataset["methods"].keys()), key="method_select")

    results_df = load_csv(dataset["methods"][method])

    query_results = results_df[results_df["qid"] == selected_qid].sort_values("rank").reset_index(drop=True)
    query_labels = labels_df[labels_df["qid"] == selected_qid].set_index("docno")["label"]
    query_results["label"] = query_results["docno"].map(query_labels)
    query_results["doc_id"] = query_results["docno"].map(chunk_to_doc)

    st.caption(f"{method} results for “{query_lookup[selected_qid]}”")

    judged = query_results.dropna(subset=["label"]).copy()
    if judged.empty:
        st.info("None of this topic's retrieved chunks have a relevance label in the pool -- nothing to rank by relevance.")
        return

    judged["label"] = judged["label"].astype(int)
    best_chunk_per_case = judged.sort_values(["label", "rank"], ascending=[False, True]).drop_duplicates(subset="doc_id")
    top_cases = best_chunk_per_case.sort_values(["label", "rank"], ascending=[False, True]).head(TOP_N)

    st.subheader(f"Top {TOP_N} case files by their most relevant chunk")
    if len(top_cases) < TOP_N:
        st.caption(f"Only {len(top_cases)} distinct labelled case file(s) among this topic's retrieved chunks.")

    for _, row in top_cases.iterrows():
        with st.container(border=True):
            header = case_display_title(row["doc_id"], case_metadata)
            st.markdown(f"#### {header}")
            st.html(
                f'<span class="label-pill">label {row["label"]}/5</span> '
                f'&nbsp;<code>doc_id: {row["doc_id"]}</code>&nbsp;&nbsp;<code>chunk: {row["docno"]}</code> (rank {row["rank"]})'
            )
            if row["doc_id"] in case_metadata.index:
                render_case_metadata(case_metadata.loc[row["doc_id"]])
            st.markdown("**Matched chunk text:**")
            st.write(row["text"] if pd.notna(row.get("text")) else "*(text unavailable)*")


def render_case_tab(
    dataset: dict,
    topics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    chunk_to_doc: Dict[str, str],
    case_metadata: pd.DataFrame,
) -> None:
    query_lookup = dict(zip(topics_df["qid"], topics_df["query_en"]))

    labelled = labels_df.copy()
    labelled["doc_id"] = labelled["docno"].map(chunk_to_doc)
    labelled = labelled.dropna(subset=["doc_id", "label"])
    labelled["label"] = labelled["label"].astype(int)

    mapped_doc_ids = sorted(labelled["doc_id"].unique())
    if not mapped_doc_ids:
        st.info("No case files have labelled chunks yet for this dataset.")
        return

    def _option_label(doc_id: str) -> str:
        title = case_display_title(doc_id, case_metadata)
        return title if len(title) <= 90 else title[:87] + "…"

    selected_doc_id = st.selectbox("Case file", options=mapped_doc_ids, format_func=_option_label, key="case_select")

    case_chunks = chunks_df[chunks_df["doc_id"] == selected_doc_id]
    if "chunk_index" in case_chunks.columns:
        case_chunks = case_chunks.sort_values("chunk_index")
    case_chunks = case_chunks.reset_index(drop=True)

    # Per chunk: {topic text -> best label for that topic}. Keyed by topic *text* (not qid) and
    # collapsed with max() up front, so a chunk matching two differently-qid'd but identically
    # named topics only ever gets one entry for that name.
    case_labels = labelled[labelled["doc_id"] == selected_doc_id]
    chunk_topics: Dict[str, Dict[str, int]] = {}
    for _, lrow in case_labels.iterrows():
        topic_text = query_lookup.get(lrow["qid"], lrow["qid"])
        entry = chunk_topics.setdefault(lrow["docno"], {})
        entry[topic_text] = max(entry.get(topic_text, -1), lrow["label"])
    chunk_best_label = {cid: max(topics.values()) for cid, topics in chunk_topics.items()}

    if not chunk_best_label:
        st.info("This case file has no labelled chunks.")
        return

    # Highlight every chunk that reached this case's own highest label -- not a fixed top-N,
    # since a short case (common in some datasets) can have very few chunks in total, and
    # capping at a fixed count made it look like only a slice of the case was being shown.
    case_max_label = max(chunk_best_label.values())
    top_chunk_ids = [cid for cid, lbl in chunk_best_label.items() if lbl == case_max_label]
    top_chunk_id_set = set(top_chunk_ids)

    header = case_display_title(selected_doc_id, case_metadata)
    st.subheader(header)
    if selected_doc_id in case_metadata.index:
        with st.container(border=True):
            render_case_metadata(case_metadata.loc[selected_doc_id])

    # One colour per distinct topic among the highlighted chunks (evenly spaced hues, in the
    # order each topic first appears) -- a chunk's own highlight colour comes from its single
    # best-labelled topic, and every topic it matches gets its own bubble in that topic's
    # colour right after the chunk text.
    ordered_topics: List[str] = []
    for cid in top_chunk_ids:
        for topic_text, _ in sorted(chunk_topics[cid].items(), key=lambda p: p[1], reverse=True):
            if topic_text not in ordered_topics:
                ordered_topics.append(topic_text)
    hue_step = 360 / max(len(ordered_topics), 1)
    topic_hue = {topic_text: round(i * hue_step) for i, topic_text in enumerate(ordered_topics)}

    st.markdown(
        f"**Highlighted:** the chunk(s) that reached this case's top label ({case_max_label}/5), "
        "coloured by topic -- each bubble names the topic(s) that chunk answers."
    )
    legend = "".join(
        f'<span class="topic-bubble" style="--hue:{topic_hue[topic_text]}">{html.escape(topic_text)}</span>'
        for topic_text in ordered_topics
    )
    st.html(f'<div style="margin-bottom: 10px;">{legend}</div>')

    html_parts = []
    for _, crow in case_chunks.iterrows():
        cid = crow["chunk_id"]
        chunk_text_html = html.escape(str(crow["chunk_text"])) if pd.notna(crow.get("chunk_text")) else ""
        if cid in top_chunk_id_set:
            pairs = sorted(chunk_topics[cid].items(), key=lambda p: p[1], reverse=True)
            best_topic_text, best_label = pairs[0]
            hue = topic_hue[best_topic_text]
            bubbles = "".join(
                f'<span class="topic-bubble" style="--hue:{topic_hue[t]}"><b>{html.escape(t)}</b> · {label}/5</span>'
                for t, label in pairs
            )
            html_parts.append(
                f'<mark class="chunk-highlight" style="--hue:{hue}">'
                f'<span class="rank-badge" style="--hue:{hue}">{best_label}/5</span>{chunk_text_html}'
                f"</mark>{bubbles}"
            )
        else:
            html_parts.append(chunk_text_html)
    doc_html = "\n\n".join(html_parts)
    st.html(f'<div class="case-doc-text">{doc_html}</div>')


def main() -> None:
    st.set_page_config(page_title="Visualise your Insights: From Case Files to Topics and Back", layout="wide")
    inject_styles()

    st.title("Visualise your Insights: From Case Files to Topics and Back")
    st.caption(
        "See which case files best answer a topic, or flip it around and see which topics "
        "a given case file answers best."
    )

    datasets = discover_datasets()
    if not datasets:
        st.error(
            "No judged datasets found. This app looks for '*_pool_labels.csv' files "
            "(produced by judge_pool.py) next to it, and needs a matching topics CSV "
            "and a known dataset's *_chunks.csv alongside them."
        )
        return

    dataset_key = st.selectbox(
        "Dataset", options=list(datasets.keys()), format_func=lambda k: datasets[k]["label"], key="dataset_select"
    )
    dataset = datasets[dataset_key]

    topics_df = with_english_topics(load_csv(dataset["topics_csv"]))
    labels_df = load_csv(dataset["labels_csv"])
    chunks_df = load_csv(dataset["chunks_csv"])
    if "doc_id" not in chunks_df.columns:
        chunks_df["doc_id"] = chunks_df["chunk_id"].apply(derive_doc_id)
    chunk_to_doc = dict(zip(chunks_df["chunk_id"], chunks_df["doc_id"]))
    case_metadata = chunks_df.drop_duplicates(subset="doc_id").set_index("doc_id")

    tab1, tab2 = st.tabs(["🔍  Topics → Case Files", "📄  Case File → Topics"])

    with tab1:
        render_query_tab(dataset, topics_df, labels_df, chunk_to_doc, case_metadata)

    with tab2:
        render_case_tab(dataset, topics_df, labels_df, chunks_df, chunk_to_doc, case_metadata)


if __name__ == "__main__":
    main()
