"""Plot the top 5 BERTopic topics per dataset as doughnut charts.

Reads each dataset's *_queries_topic_names.csv (produced by extract_topics.py),
excludes the -1 "Outliers" bucket (not a real topic), takes the top 5 by
num_queries, and saves one doughnut chart per dataset as its own (small) PNG,
with topic names written directly inside each wedge (no legend).

Usage:

    python3 plot_top_topics.py
    python3 plot_top_topics.py --output-prefix plots/top_topics
"""

from __future__ import annotations

import argparse
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATASETS = {
    "euaa": "data/euaa_asylum_report_queries_topic_names.csv",
    "fln": "data/fln_praksis_2026_queries_topic_names.csv",
    "asylex": "data/asylex_raw_documents_sample_queries_topic_names.csv",
}

TOP_N = 5

# seaborn's "pastel" palette, used by *rank* (1st, 2nd, ... 5th most frequent
# topic) so the same colour means the same rank in every dataset's chart.
RANK_COLORS = sns.color_palette("pastel", n_colors=TOP_N)

# Danish topic names get translated to English; every name is then title-cased
# for consistent display (matches the summary table shown alongside this chart).
TOPIC_NAME_OVERRIDES = {
    "flygtningenævnet asylum decisions": "Refugee Appeals Board Asylum Decisions",
}


def display_name(name: str) -> str:
    return TOPIC_NAME_OVERRIDES.get(name, name.title())


def load_top_topics(path: str, top_n: int = TOP_N) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["topic_id"] != -1]  # drop the "Outliers" noise bucket, not a real topic
    top = df.sort_values("num_queries", ascending=False).head(top_n).copy()
    top["topic_name"] = top["topic_name"].apply(display_name)
    return top


def plot_one(dataset: str, path: str, output_path: str) -> None:
    top = load_top_topics(path)  # sorted descending: row 0 = rank 1
    values = top["num_queries"].tolist()
    names = top["topic_name"].tolist()
    colors = RANK_COLORS[: len(values)]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.5),
    )

    # Placed by hand (rather than via ax.pie(labels=...)) so a narrow wedge's
    # font size and text-wrap width shrink with its angular width -- keeps
    # each label inside its own wedge instead of spilling into its neighbours.
    # Consecutive small wedges (a tightly bunched run) also get their labels
    # fanned out across a few different radii, so those labels don't merge
    # into each other the way they would all sitting at the same radius.
    radius_cycle = [0.78, 1.05, 1.32]
    total = sum(values)
    cursor = 90.0
    run_idx = 0
    for value, name in zip(values, names):
        sweep = value / total * 360
        mid_angle = cursor - sweep / 2
        cursor -= sweep

        if sweep < 45:
            radius = radius_cycle[run_idx % len(radius_cycle)]
            run_idx += 1
        else:
            radius = 0.78
            run_idx = 0

        fontsize = 12
        label = f"{textwrap.fill(name, 14, break_long_words=False)}\n({value})"

        x = np.cos(np.radians(mid_angle)) * radius
        y = np.sin(np.radians(mid_angle)) * radius
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#333333", linespacing=0.9)

    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


def plot(output_prefix: str) -> None:
    sns.set_theme(style="white", font_scale=1.0)
    for dataset, path in DATASETS.items():
        plot_one(dataset, path, f"{output_prefix}_{dataset}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="plots/top_topics", help="Output filename prefix (writes '<prefix>_<dataset>.png' per dataset)")
    args = parser.parse_args()
    plot(args.output_prefix)


if __name__ == "__main__":
    main()
