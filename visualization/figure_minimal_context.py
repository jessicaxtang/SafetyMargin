#!/usr/bin/env python3
"""
Produce the Minimal vs Full Context margin comparison (formerly Figure A3).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from figure_common import (
    SUITE_PALETTE,
    SUITES,
    prepare_suite_payload,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)


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
        help="Directory containing per-run JSON artefacts",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name to filter within span metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fig/analysis/figure_MinimalContext.png"),
        help="Output path for the margin comparison figure",
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
        default=8.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.5,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Reserved for compatibility; no randomness used presently",
    )
    return parser.parse_args()


def build_line_data(suite_payload: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, suite in enumerate(SUITES):
        payload = suite_payload.get(suite.key)
        if payload is None:
            continue
        base_scores = payload.get("base_scores") or []
        for entry_idx, scores in enumerate(base_scores):  # type: ignore[assignment]
            m_min = scores["m_minimal"]
            m_full = scores["m_full"]
            run_id = scores.get("run_id", entry_idx)
            pair_id = f"{suite.key}_{run_id}_{entry_idx}"
            rows.append(
                {
                    "suite_label": suite.label,
                    "x": idx - 0.18,
                    "margin": m_min,
                    "baseline": "Minimal",
                    "pair_id": pair_id,
                }
            )
            rows.append(
                {
                    "suite_label": suite.label,
                    "x": idx + 0.18,
                    "margin": m_full,
                    "baseline": "Minimal+Spec",
                    "pair_id": pair_id,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    suite_payload = prepare_suite_payload(
        args.span_metrics,
        model_name=args.model_name,
        top_k=4,
        include_base_scores=True,
        results_dir=args.results_dir,
    )

    data = build_line_data(suite_payload)
    if data.empty:
        raise ValueError("No base score comparisons found for the requested configuration.")

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    sns.lineplot(
        data=data,
        x="x",
        y="margin",
        hue="suite_label",
        units="pair_id",
        estimator=None,
        marker="o",
        linewidth=1.3,
        palette=SUITE_PALETTE,
        ax=ax,
    )
    try:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="Suite")
    except AttributeError:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="Suite")

    y_min = float(data["margin"].min())
    y_max = float(data["margin"].max())
    y_pad = (y_max - y_min) * 0.18 if y_max > y_min else 0.2
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(-0.6, len(SUITES) - 0.4)
    ax.set_xticks(range(len(SUITES)))
    ax.set_xticklabels([suite.label for suite in SUITES])
    ax.set_ylabel(r"$M(x)$ margin")
    ax.set_xlabel("Suite")
    ax.set_title("Minimal vs Full Context margins (per suite)")
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=10)

    suite_full_max: Dict[str, float] = {}
    for suite in SUITES:
        payload = suite_payload.get(suite.key)
        if payload is None:
            continue
        base_scores = payload.get("base_scores") or []
        for scores in base_scores:
            suite_full_max[suite.key] = max(
                suite_full_max.get(suite.key, float("-inf")),
                float(scores["m_full"]),
            )

    for idx, suite in enumerate(SUITES):
        payload = suite_payload.get(suite.key)
        if not payload:
            continue
        deltas = []
        for scores in payload.get("base_scores") or []:
            deltas.append(float(scores["m_full"]) - float(scores["m_minimal"]))
        if not deltas:
            continue
        med_delta = float(np.median(deltas))
        max_full = suite_full_max.get(suite.key)
        if max_full is None or not np.isfinite(max_full):
            continue
        ax.text(
            idx,
            max_full + y_pad * 0.35,
            f"Δ̃={med_delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=SUITE_PALETTE.get(suite.key, "#333333"),
        )

    ax.text(
        -0.08,
        1.05,
        "MinimalContext",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote minimal vs full context figure to {output_path}")


if __name__ == "__main__":
    main()
