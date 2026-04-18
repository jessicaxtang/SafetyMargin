#!/usr/bin/env python3
"""Render a color-coded view of a single handcrafted prompt example.

For each prompt unit in the requested example, we show:
- The unit text.
- ΔM_LOO = M_removed - M_full (margin change under leave-one-out).
- ΔM_AOI (add-one-in delta when adding only that unit back).

LOO cells are shown on the left column, AOI on the right. Positive (helpful)
values are shaded green, negative (harmful) values are shaded red, with the
intensity scaled by the absolute margin magnitude.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import textwrap
import numpy as np

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-unit-path",
        type=Path,
        # required=True,
        default=Path("experiments-helpful/reference_attribution_n100_seed42/per_unit_rows.csv"),
        help="per_unit_rows.csv produced by reference_margin_attribution.py",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        required=True,
        help="Example index to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualization/out/example_span_colormap.png"),
        help="PNG file to write.",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help="Optional max absolute margin for normalization (defaults to max observed).",
    )
    return parser.parse_args()


def load_example_rows(per_unit_path: Path, example_index: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with per_unit_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row.get("example_index", "nan"))) != example_index:
                continue
            if row.get("unit_kind") != "prompt_unit":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompt units found for example_index={example_index}")
    rows.sort(key=lambda r: int(float(r.get("unit_index", 0))))
    return rows


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def blend_with_white(base_color: str, intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    base = _hex_to_rgb(base_color)
    blended = tuple(int(255 - (255 - component) * intensity) for component in base)
    return _rgb_to_hex(blended)


def color_for_value(value: Optional[float], max_abs: float) -> Tuple[str, str]:
    if value is None or max_abs <= 0:
        return "#f8f9fa", "#495057"
    if value == 0:
        return "#e9ecef", "#495057"
    base_color = "#2ca02c" if value > 0 else "#d62728"
    intensity = min(abs(value) / max_abs, 1.0)
    bg = blend_with_white(base_color, intensity)
    fg = "#0f5132" if value > 0 else "#842029"
    return bg, fg


def fmt_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:+.3f}"


def parse_loo_delta(row: Dict[str, str]) -> float:
    """Return LOO delta as margin change from removing a unit: M_removed - M_full."""
    delta_ref = row.get("delta_ref")
    if delta_ref not in (None, "", "None"):
        return float(delta_ref)

    attribution = row.get("attribution")
    if attribution in (None, "", "None"):
        raise ValueError("Row is missing both delta_ref and attribution for LOO value.")
    return -float(attribution)


def build_figure(rows: List[Dict[str, str]], output_path: Path, max_abs_override: Optional[float]) -> None:
    loo_values = [parse_loo_delta(row) for row in rows]
    aoi_values = [
        float(row["aoi_delta"])
        for row in rows
        if row.get("aoi_delta") not in (None, "", "None")
    ]
    if not loo_values and not aoi_values:
        raise ValueError("No margin values found for the selected example.")
    max_abs = max([abs(v) for v in loo_values + aoi_values])
    if max_abs_override is not None and max_abs_override > 0:
        max_abs = max_abs_override
    if max_abs <= 0:
        max_abs = 1.0

    labels = [textwrap.fill(row.get("unit_text", ""), width=38) for row in rows]
    loo = [parse_loo_delta(row) for row in rows]
    aoi = [
        float(row["aoi_delta"]) if row.get("aoi_delta") not in (None, "", "None") else 0.0
        for row in rows
    ]

    n = len(rows)
    y_positions = np.arange(n, dtype=float)
    fig_height = max(3.5, 0.85 * n + 1.0)
    fig = plt.figure(figsize=(12, fig_height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.15)
    text_ax = fig.add_subplot(gs[0, 0])
    loo_ax = fig.add_subplot(gs[0, 1], sharey=text_ax)
    aoi_ax = fig.add_subplot(gs[0, 2], sharey=text_ax)

    # Text column
    text_ax.set_xlim(0, 1)
    text_ax.set_ylim(-0.5, n - 0.5)
    text_ax.axis("off")
    for y, label in zip(y_positions, labels):
        text_ax.text(1.0, y, label, ha="right", va="center", fontsize=12, color="#212529")

    titles = ["Leave-one-out ΔM (removed - full)", "Add-one-in ΔM"]
    axes = [loo_ax, aoi_ax]
    datasets = [loo, aoi]
    for ax, values, title in zip(axes, datasets, titles):
        if ax is loo_ax:
            # For LOO = removed - full, positive means removing helped (unit was harmful).
            colors = ["#f97272" if v >= 0 else "#3f9b57" for v in values]
        else:
            colors = ["#3f9b57" if v >= 0 else "#f97272" for v in values]
        ax.barh(y_positions, values, color=colors, edgecolor="none", height=0.6)
        ax.axvline(0, color="#4a4a4a", linewidth=1.2)
        ax.set_xlim(-max_abs * 1.05, max_abs * 1.05)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.9)
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Δ margin", fontsize=11)

    for ax in axes + [text_ax]:
        ax.invert_yaxis()
    loo_ax.set_yticks([])
    aoi_ax.set_yticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote color-coded example visualization to {output_path}")


def main() -> None:
    args = parse_args()
    rows = load_example_rows(args.per_unit_path, args.example_index)
    output_path = Path(f"{args.output}_example{args.example_index}.png")
    build_figure(rows, output_path, args.max_abs)


if __name__ == "__main__":
    main()
