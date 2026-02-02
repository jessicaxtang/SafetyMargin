#!/usr/bin/env python3
"""
Render a heatmap of pairwise span interaction deltas (ΔM) from SafetyMargin analysis.

Example:
  python visualization/plot_interaction_heatmap.py \
      --interactions analysis/pairwise_interactions_full.csv \
      --run-id privacy_run20251104T224829 \
      --output fig/analysis/interaction_heatmap.png
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", font_scale=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("analysis/pairwise_interactions_full.csv"),
        help="CSV produced by scripts/analyze_results.py describing pairwise interaction deltas.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional filter for model_name.",
    )
    parser.add_argument(
        "--suite",
        help="Optional filter for suite key (e.g. evidence, privacy).",
    )
    parser.add_argument(
        "--run-id",
        help="Optional filter for a specific run identifier.",
    )
    parser.add_argument(
        "--case-id",
        help="Optional filter for a specific case identifier.",
    )
    parser.add_argument(
        "--top-k-spans",
        type=int,
        default=10,
        help="Limit to the top-K spans by absolute interaction magnitude (per axis).",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=0.0,
        help="Discard interaction entries with |delta| below this threshold.",
    )
    parser.add_argument(
        "--annot",
        action="store_true",
        help="Display numeric annotations on the heatmap.",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap to use.",
    )
    parser.add_argument(
        "--title",
        help="Optional title for the figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fig/analysis/interaction_heatmap.png"),
        help="Destination path for the rendered heatmap.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {kind} at {path} but it was not found.")


def format_span(text: str, width: int = 40) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    return textwrap.fill(textwrap.shorten(cleaned, width=width, placeholder="…"), width=width // 2 or 20)


def load_interactions(
    csv_path: Path,
    *,
    model_name: Optional[str],
    suite: Optional[str],
    run_id: Optional[str],
    case_id: Optional[str],
) -> pd.DataFrame:
    ensure_exists(csv_path, "interaction CSV")
    df = pd.read_csv(csv_path)
    if model_name:
        df = df[df["model_name"] == model_name]
    if suite:
        df = df[df["suite"] == suite]
    if run_id:
        df = df[df["run_id"] == run_id]
    if case_id:
        df = df[df["case_id"] == case_id]
    return df


def select_span_subset(df: pd.DataFrame, top_k: int) -> Sequence[str]:
    if top_k <= 0:
        spans_i = df["span_i_text"].unique()
        spans_j = df["span_j_text"].unique()
        return sorted(set(spans_i) | set(spans_j))
    # Compute aggregated strength per span across both roles.
    strength = {}
    for col in ("span_i_text", "span_j_text"):
        grouped = df.groupby(col)["delta"].apply(lambda series: float(np.nanmax(np.abs(series))))
        strength.update(grouped.to_dict())
    top_spans = sorted(strength.items(), key=lambda item: item[1], reverse=True)[: top_k * 2]
    return [span for span, _ in top_spans]


def build_heatmap_data(
    df: pd.DataFrame,
    *,
    top_k_spans: int,
    abs_threshold: float,
) -> pd.DataFrame:
    df = df.copy()
    df = df[np.abs(df["delta"]) >= abs_threshold].copy()
    if df.empty:
        raise ValueError("No interaction rows remain after applying filters/thresholds.")

    span_subset = select_span_subset(df, top_k_spans)
    df = df[df["span_i_text"].isin(span_subset) & df["span_j_text"].isin(span_subset)].copy()
    if df.empty:
        raise ValueError("No interactions remain after limiting to top spans; relax --top-k-spans.")

    df["span_i_label"] = df["span_i_text"].apply(format_span)
    df["span_j_label"] = df["span_j_text"].apply(format_span)
    df["abs_delta"] = df["delta"].abs()

    # Aggregate duplicate pairs (using mean delta).
    pivot = (
        df.pivot_table(
            index="span_i_label",
            columns="span_j_label",
            values="delta",
            aggfunc="mean",
            fill_value=0.0,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # Reorder axes by overall strength for readability.
    row_strength = pivot.abs().max(axis=1).sort_values(ascending=False)
    col_strength = pivot.abs().max(axis=0).sort_values(ascending=False)
    pivot = pivot.loc[row_strength.index, col_strength.index]

    return pivot


def main() -> None:
    args = parse_args()
    df_raw = load_interactions(
        args.interactions,
        model_name=args.model_name,
        suite=args.suite,
        run_id=args.run_id,
        case_id=args.case_id,
    )
    if df_raw.empty:
        raise ValueError("No interaction rows matched the provided filters.")

    heatmap_df = build_heatmap_data(
        df_raw,
        top_k_spans=args.top_k_spans,
        abs_threshold=args.abs_threshold,
    )

    plt.figure(figsize=(max(6, heatmap_df.shape[1] * 0.6), max(5, heatmap_df.shape[0] * 0.5)))
    ax = sns.heatmap(
        heatmap_df,
        annot=args.annot,
        fmt=".2f",
        cmap=args.cmap,
        center=0.0,
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_kws={"label": r"Interaction $\Delta M$"},
    )
    ax.set_xlabel("Span j")
    ax.set_ylabel("Span i")

    title_parts: List[str] = []
    if args.title:
        title_parts.append(args.title)
    else:
        if args.case_id:
            title_parts.append(args.case_id)
        if args.run_id:
            title_parts.append(args.run_id)
        if args.suite:
            title_parts.append(args.suite.title())
    if title_parts:
        ax.set_title(" | ".join(title_parts), pad=14)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote interaction heatmap to {args.output}")


if __name__ == "__main__":
    main()
