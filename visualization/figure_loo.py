#!/usr/bin/env python3
"""
Produce span-level ΔM bar plots for Leave-One-Out (LOO), AOI, or AOI intent metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from figure_common import (
    SUITES,
    prepare_suite_payload,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", font_scale=0.95, font="Sans Serif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--span-metrics",
        type=Path,
        default=Path("analysis/span_metrics_full.csv"),
        help="CSV produced by scripts/analyze_results.py",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/meta-llama_Llama-3.2-1B-Instruct"),
        help="Directory containing per-run JSON artefacts (needed for intent AOI plots).",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name to filter within span metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fig/analysis/figure_LOO.png"),
        help="Output path for the combined panel figure",
    )
    parser.add_argument(
        "--metric",
        choices=("loo", "aoi", "aoi_intent"),
        default="loo",
        help="Metric to visualise (LOO ΔM, AOI ΔM, or AOI ΔM grouped by intent).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of spans per suite to show",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure save DPI",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=2.8,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--single-width",
        type=float,
        default=4.5, #2.5
        help="Figure width for individual suite panels",
    )
    parser.add_argument(
        "--single-height",
        type=float,
        default=3.5,
        help="Figure height for individual suite panels",
    )
    parser.add_argument(
        "--x-label-size",
        type=int,
        default=8,
        help="Font size for x-axis tick labels",
    )
    # Threshold controls
    parser.add_argument(
        "--tau-risk",
        type=float,
        default=0.20,
        help="Decision threshold for |LOO| used for vertical guide lines.",
    )
    parser.add_argument(
        "--tau-protect",
        type=float,
        default=0.20,
        help="Decision threshold for |AOI| used for vertical guide lines.",
    )
    parser.add_argument(
        "--tau-intentntent",
        type=float,
        default=0.15,
        help="Decision threshold for |AOI intent| used for vertical guide lines.",
    )
    parser.add_argument(
        "--hide-thresholds",
        action="store_true",
        help="Do not draw vertical decision threshold guide lines.",
    )
    parser.add_argument(
        "--hide-titles",
        action="store_true",
        help="Disable suite titles that include the selected run id",
    )
    return parser.parse_args()


def adjust_stem_for_metric(stem: str, metric: str) -> str:
    replacements = {"aoi": "AOI", "aoi_intent": "AOI_INTENT"}
    replacement = replacements.get(metric)
    if not replacement:
        return stem
    repl_lower = replacement.lower()
    if "LOO" in stem:
        return stem.replace("LOO", replacement)
    if "loo" in stem:
        return stem.replace("loo", repl_lower)
    if metric == "aoi_intent":
        if "AOI" in stem:
            return stem.replace("AOI", replacement)
        if "aoi" in stem:
            return stem.replace("aoi", repl_lower)
    if replacement in stem or repl_lower in stem:
        return stem
    return f"{stem}_{replacement}"


def adjust_output_path(path: Path, metric: str) -> Path:
    if metric == "loo":
        return path
    suffix = path.suffix or ".png"
    new_stem = adjust_stem_for_metric(path.stem, metric)
    if new_stem == path.stem:
        return path
    return path.with_name(f"{new_stem}{suffix}")


def select_metric_rows(
    payload: Dict[str, object],
    metric: str,
    top_k: int,
):
    import pandas as pd  # type: ignore

    if metric == "loo":
        # top_spans = payload.get("top_spans")
        # TO DO: ascending = True
        top_spans = payload.get("top_spans")
        if top_spans is not None:
            top_spans = top_spans.sort_values("abs_loo", ascending=True)

        if top_spans is None or getattr(top_spans, "empty", True):
            return None
        if top_k > 0:
            return top_spans.head(top_k)
        return top_spans

    metric_col = "aoi_minimal"
    if metric == "aoi_intent":
        source = payload.get("intent_aoi")
        if not isinstance(source, pd.DataFrame) or source.empty or metric_col not in source.columns:
            return None
        df = source.dropna(subset=[metric_col]).copy()
        if df.empty:
            return None
        abs_col = f"abs_{metric}"
        df[abs_col] = df[metric_col].abs()
        if top_k > 0:
            df = df.sort_values(abs_col, ascending=True).head(top_k)
        else:
            df = df.sort_values(abs_col, ascending=True)
        return df

    # For AOI minimal: keep the same span set and order as LOO for consistency
    source = payload.get("suite_df")
    if not isinstance(source, pd.DataFrame) or source.empty or metric_col not in source.columns:
        return None

    # Build LOO-ordered span labels
    loo_df = payload.get("top_spans")
    if loo_df is not None and not getattr(loo_df, "empty", True):
        try:
            loo_sorted = loo_df.sort_values("abs_loo", ascending=True)
        except Exception:
            loo_sorted = loo_df
        if top_k > 0:
            loo_sorted = loo_sorted.head(top_k)
        ordered_labels = list(loo_sorted["span_label"].astype(str).tolist())

        # Filter AOI rows to these labels and preserve order via categorical
        df = source.dropna(subset=[metric_col]).copy()
        df = df[df["span_label"].astype(str).isin(ordered_labels)].copy()
        if df.empty:
            return None
        df["_order"] = pd.Categorical(df["span_label"].astype(str), categories=ordered_labels, ordered=True)
        df = df.sort_values("_order").drop(columns=["_order"])  # ordered to match LOO
        return df

    # Fallback: order by |AOI| if no LOO info is available
    df = source.dropna(subset=[metric_col]).copy()
    if df.empty:
        return None
    abs_col = f"abs_{metric}"
    df[abs_col] = df[metric_col].abs()
    if top_k > 0:
        df = df.sort_values(abs_col, ascending=True).head(top_k)
    else:
        df = df.sort_values(abs_col, ascending=True)
    return df


def draw_suite_panel(
    ax: plt.Axes,
    suite,
    payload: Optional[Dict[str, object]],
    *,
    show_title: bool,
    panel_label: Optional[str] = None,
    x_label_size: int = 8,
    metric: str,
    top_k: int,
    threshold: Optional[float],
    show_thresholds: bool,
) -> bool:
    if not payload:
        ax.set_axis_off()
        return False
    top_spans = select_metric_rows(payload, metric, top_k)
    if top_spans is None or getattr(top_spans, "empty", True):
        ax.set_axis_off()
        return False

    metric_col = "loo" if metric == "loo" else "aoi_minimal"
    values = top_spans[metric_col].to_numpy(dtype=float)
    labels: Sequence[str] = list(top_spans["span_label"].to_numpy())

    # LOO COLOURS (default): positive spans (risk-increasing), negative (protective)
    POSITIVE_COLOR = "#ff8b8b" # red
    NEGATIVE_COLOR = "#42a73f" # green

    if metric in ("aoi", "aoi_intent"):
        POSITIVE_COLOR = "#42a73f" # green
        NEGATIVE_COLOR = "#ff8b8b" # red
    palette = {
        label: (POSITIVE_COLOR if val >= 0 else NEGATIVE_COLOR)
        for label, val in zip(labels, values)
    }
    sns.barplot(
        data=top_spans,
        x=metric_col,
        y="span_label",
        order=labels,
        orient="h",
        palette=palette,
        errorbar=None,
        ax=ax,
    )
    # Remove bar outlines for a cleaner flat look
    for patch in ax.patches:
        # use 'none' so bars have no visible edge; keep facecolor from seaborn palette
        patch.set_edgecolor("none")
        patch.set_linewidth(0.0)
    ax.set_ylabel("")
    ax.set_xlabel(r"LOO $\Delta M$" if metric == "loo" else r"AOI $\Delta M$", fontsize=14) # CHANGE THIS FOR X AXIS LABEL SIZE
    # Zero reference line
    ax.axvline(0.0, color="#555555", linewidth=1.0)
    # Optional decision threshold guide lines at ±threshold
    if show_thresholds and threshold is not None and np.isfinite(threshold) and threshold > 0:
        ax.axvline(
            float(threshold),
            color="#888888",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.axvline(
            -float(threshold),
            color="#888888",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
    if show_title:
        title_run = payload.get("run_id") or "n/a"
        ax.set_title(f"{suite.label}\n{title_run}", fontsize=11, pad=10)
    ax.tick_params(axis="y", labelsize=12) # CHANGE THIS FOR Y TICKS LABEL SIZE
    ax.tick_params(axis="x", labelsize=x_label_size)
    ax.grid(axis="x", alpha=0.2)
    limit = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    # Ensure thresholds are visible within axis limits
    if show_thresholds and threshold is not None and np.isfinite(threshold) and threshold > 0:
        limit = max(limit, float(threshold))
    ax.set_xlim(-limit * 1.1, limit * 1.1)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=False)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    if panel_label:
        ax.text(
            -0.35,
            1.05,
            panel_label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )
    return True


def plot_suite_axes(
    fig: plt.Figure,
    suite_payload: Dict[str, Dict[str, object]],
    show_titles: bool,
    x_label_size: int,
    metric: str,
    top_k: int,
    threshold: Optional[float],
    show_thresholds: bool,
) -> List[plt.Axes]:
    grid = GridSpec(1, len(SUITES), figure=fig, wspace=0.32)
    axes: List[plt.Axes] = []
    for idx, suite in enumerate(SUITES):
        ax = fig.add_subplot(grid[0, idx])
        axes.append(ax)
        draw_suite_panel(
            ax,
            suite,
            suite_payload.get(suite.key),
            show_title=show_titles,
            panel_label=None, #"LOO" if idx == 0 else None,
            x_label_size=x_label_size,
            metric=metric,
            top_k=top_k,
            threshold=threshold,
            show_thresholds=show_thresholds,
        )
    return axes


def save_individual_figures(
    suite_payload: Dict[str, Dict[str, object]],
    base_output: Path,
    *,
    dpi: int,
    show_titles: bool,
    width: float,
    height: float,
    x_label_size: int,
    metric: str,
    top_k: int,
    threshold: Optional[float],
    show_thresholds: bool,
) -> None:
    parent = base_output.parent
    parent.mkdir(parents=True, exist_ok=True)
    base_name_metric = adjust_stem_for_metric(base_output.stem, metric)
    suffix = base_output.suffix or ".png"

    saved_paths: List[Path] = []
    for suite in SUITES:
        payload = suite_payload.get(suite.key)
        fig, ax = plt.subplots(figsize=(width, height))
        rendered = draw_suite_panel(
            ax,
            suite,
            payload,
            show_title=show_titles,
            panel_label=None, #"LOO",
            x_label_size=x_label_size,
            metric=metric,
            top_k=top_k,
            threshold=threshold,
            show_thresholds=show_thresholds,
        )
        if not rendered:
            plt.close(fig)
            continue
        fig.tight_layout()
        out_path = parent / f"{base_name_metric}_{suite.key}{suffix}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    if saved_paths:
        print("Saved individual suite figures:")
        for path in saved_paths:
            print(f"  - {path}")


def main() -> None:
    args = parse_args()
    include_intent = args.metric == "aoi_intent"
    results_dir = args.results_dir if include_intent else None
    # Select the appropriate decision threshold for the chosen metric
    if args.metric == "loo":
        threshold_val: Optional[float] = float(args.tau_risk)
    elif args.metric == "aoi":
        threshold_val = float(args.tau_protect)
    else:
        threshold_val = float(args.tau_intent)
    suite_payload = prepare_suite_payload(
        args.span_metrics,
        model_name=args.model_name,
        top_k=args.top_k,
        include_intent_aoi=include_intent,
        results_dir=results_dir,
    )
    if include_intent:
        has_intent_data = any(
            payload.get("intent_aoi") is not None for payload in suite_payload.values()
        )
        if not has_intent_data:
            print(
                "[figure_loo] Warning: No AOI intent data found for the selected runs. "
                "Proceeding without intent panels; check that --span-metrics run_ids "
                "match JSONs in --results-dir or adjust --results-dir."
            )

    fig = plt.figure(figsize=(args.width, args.height))
    plot_suite_axes(
        fig,
        suite_payload,
        show_titles=not args.hide_titles,
        x_label_size=args.x_label_size,
        metric=args.metric,
        top_k=args.top_k,
        threshold=threshold_val,
        show_thresholds=not args.hide_thresholds,
    )
    fig.tight_layout()

    output_path: Path = adjust_output_path(args.output, args.metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.metric.upper()} ΔM figure to {output_path}")
    save_individual_figures(
        suite_payload,
        output_path,
        dpi=args.dpi,
        show_titles=not args.hide_titles,
        width=args.single_width,
        height=args.single_height,
        x_label_size=args.x_label_size,
        metric=args.metric,
        top_k=args.top_k,
        threshold=threshold_val,
        show_thresholds=not args.hide_thresholds,
    )


if __name__ == "__main__":
    main()
