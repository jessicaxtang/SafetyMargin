#!/usr/bin/env python3
"""
Plot "Deletion Faithfulness (LOO)" curves from the SafetyMargin pipeline JSON results.

Layout:
- Rows: test cases (Privacy, Injection, Role Confusion, Safety, Evidence)
- Columns: variants (Baseline, Fixed, Final)
- Each subplot: cumulative sum of LOO (margin-space) when removing spans in LOO rank
  order vs a length-matched random baseline.

Design:
- X-axis: Top-k spans removed (1..K)
- Y-axis: Cumulative ΔM (higher is better)
- Two lines: LOO order (solid, thicker) and Random (dashed)
- If multiple runs present for a variant/case, plot mean with 95% CI ribbon
- Consistent y-limits across all subplots
- Annotate steepest drop point with a span label

Usage example:
  python visualization/plot_deletion_faithfulness.py \
    --model meta-llama_Llama-3.2-1B-Instruct \
    --results-root results \
    --output-dir fig
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_NAMES = [
    ("privacy", "Privacy"),
    ("injection", "Injection"),
    ("role_confusion", "Role Confusion"),
    ("safety", "Safety"),
    ("evidence", "Evidence"),
]

VARIANTS = [
    ("Baseline", "_run"),
    ("Fixed", "_fixed_run"),
    # ("Final", "_final_run"),
]


@dataclass
class RunBundle:
    path: Path
    spans: List[str]
    loo: List[float]
    ranked_indices: List[int]


def find_runs(results_dir: Path) -> Dict[str, Dict[str, List[RunBundle]]]:
    """Scan results directory and group runs by case and variant.

    Returns: case_key -> variant_name -> list[RunBundle]
    """
    groups: Dict[str, Dict[str, List[RunBundle]]] = {k: {v[0]: [] for v in VARIANTS} for k, _ in CASE_NAMES}
    for path in sorted(results_dir.glob("*.json")):
        name = path.name.lower()
        for case_key, _ in CASE_NAMES:
            if not name.startswith(case_key):
                continue
            for variant_name, token in VARIANTS:
                if token in name:
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            payload = json.load(f)
                        spans = list(payload.get("spans") or [])
                        loo_sec = payload.get("loo") or {}
                        loo_vals = [float(x) for x in (loo_sec.get("loo") or [])]
                        ranked = loo_sec.get("ranked_indices")
                        if ranked is None:
                            # Fallback to abs order
                            ranked = sorted(range(len(loo_vals)), key=lambda i: abs(loo_vals[i]), reverse=True)
                        ranked = [int(i) for i in ranked]
                        if not loo_vals:
                            continue
                        groups[case_key][variant_name].append(
                            RunBundle(path=path, spans=spans, loo=loo_vals, ranked_indices=ranked)
                        )
                    except Exception:
                        continue
    return groups


def cumulative_curve_in_rank_order(bundle: RunBundle) -> np.ndarray:
    order = [i for i in bundle.ranked_indices if 0 <= i < len(bundle.loo)]
    values = np.array([bundle.loo[i] for i in order], dtype=float)
    return np.cumsum(values)


def random_baseline_curves(bundle: RunBundle, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = len(bundle.loo)
    if n <= 0 or n_samples <= 0:
        return np.zeros((0, 0), dtype=float)
    loo_vals = np.array(bundle.loo, dtype=float)
    curves = np.zeros((n_samples, n), dtype=float)
    for s in range(n_samples):
        perm = rng.permutation(n)
        vals = loo_vals[perm]
        curves[s, :] = np.cumsum(vals)
    return curves


def mean_and_ci(data: np.ndarray, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and lower/upper pointwise percentile bands along axis 0.
    Expects data shape (S, K).
    """
    if data.size == 0:
        return np.array([]), np.array([]), np.array([])
    alpha = (1.0 - ci) / 2.0
    lower = np.nanpercentile(data, alpha * 100.0, axis=0)
    upper = np.nanpercentile(data, (1.0 - alpha) * 100.0, axis=0)
    mean = np.nanmean(data, axis=0)
    return mean, lower, upper


def annotate_steepest(ax: plt.Axes, bundle: RunBundle, curve: np.ndarray) -> None:
    if curve.size == 0:
        return
    # Find steepest negative step in ranked order for this bundle
    order = [i for i in bundle.ranked_indices if 0 <= i < len(bundle.loo)]
    steps = np.array([bundle.loo[i] for i in order], dtype=float)
    if steps.size == 0:
        return
    idx = int(np.argmin(steps))
    x = idx + 1
    y = curve[idx]
    label = ""
    if 0 <= order[idx] < len(bundle.spans):
        label = str(bundle.spans[order[idx]]).strip()
        label = label.replace("\n", " ")
        if len(label) > 60:
            label = label[:57] + "…"
    if label:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + max(1, len(curve)//10), y),
            textcoords="data",
            fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
            va="center",
        )


def plot_grid(groups: Dict[str, Dict[str, List[RunBundle]]], output_dir: Path, title: str, n_random: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)

    # Prepare data aggregates per cell
    grid_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}
    global_min = +float("inf")
    global_max = -float("inf")

    for r, (case_key, case_label) in enumerate(CASE_NAMES):
        for c, (variant_name, _token) in enumerate(VARIANTS):
            runs = groups.get(case_key, {}).get(variant_name) or []
            if not runs:
                grid_stats[(r, c)] = {"present": False}
                continue

            # LOO curves in rank order (one per run)
            loo_curves = [cumulative_curve_in_rank_order(b) for b in runs]
            # Normalize lengths to min common K (to align across runs)
            K = min((len(curve) for curve in loo_curves), default=0)
            loo_curves = [curve[:K] for curve in loo_curves if K > 0]

            # Random baseline: pool random curves across runs
            rand_curves_all: List[np.ndarray] = []
            for b in runs:
                rand_curves_all.append(random_baseline_curves(b, n_random, rng))
            rand_curves = np.vstack([cur[:, :K] for cur in rand_curves_all if cur.size > 0]) if K > 0 else np.zeros((0, 0))

            # Stats
            loo_mat = np.vstack(loo_curves) if loo_curves else np.zeros((0, 0))
            loo_mean, loo_lo, loo_hi = mean_and_ci(loo_mat, ci=0.95)
            rnd_mean, rnd_lo, rnd_hi = mean_and_ci(rand_curves, ci=0.95)

            # Track global y-limits
            for arr in (loo_lo, loo_hi, rnd_lo, rnd_hi, loo_mean, rnd_mean):
                if arr.size:
                    global_min = min(global_min, float(np.nanmin(arr)))
                    global_max = max(global_max, float(np.nanmax(arr)))

            # Choose a representative run for annotation preference: Final > Fixed > Baseline
            preferred = None
            if variant_name == "Final":
                preferred = runs[0]
            elif variant_name == "Fixed":
                preferred = runs[0]
            else:
                preferred = runs[0]

            grid_stats[(r, c)] = {
                "present": True,
                "K": K,
                "loo_mean": loo_mean,
                "loo_lo": loo_lo,
                "loo_hi": loo_hi,
                "rnd_mean": rnd_mean,
                "rnd_lo": rnd_lo,
                "rnd_hi": rnd_hi,
                "rep_bundle": preferred,
                "case_label": case_label,
                "variant_label": variant_name,
            }

    # Fallback limits
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = -1.0, 1.0
    if abs(global_max - global_min) < 1e-6:
        pad = 0.5
    else:
        pad = 0.05 * (global_max - global_min)
    y_min = global_min - pad
    y_max = global_max + pad

    # Per-row figures instead of one big grid
    n_cols = len(VARIANTS)
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    for r, (case_key, case_label) in enumerate(CASE_NAMES):
        fig, axes = plt.subplots(1, n_cols, figsize=(8.0 * n_cols, 3.6), sharex=False, sharey=False)
        if n_cols == 1:
            axes = np.array([axes])
        for c, (variant_name, _token) in enumerate(VARIANTS):
            ax = axes[c]
            stats = grid_stats.get((r, c), {"present": False})
            ax.set_title(f"{variant_name}")
            ax.axhline(0.0, color="#888", lw=0.8)
            ax.grid(True, axis="y", alpha=0.15)
            if not stats.get("present"):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#777")
                ax.set_xlim(0, 1)
                ax.set_ylim(-1, 1)
                continue

            K = int(stats["K"] or 0)
            x = np.arange(1, K + 1)
            loo_mean = stats["loo_mean"]
            loo_lo = stats["loo_lo"]
            loo_hi = stats["loo_hi"]
            rnd_mean = stats["rnd_mean"]
            rnd_lo = stats["rnd_lo"]
            rnd_hi = stats["rnd_hi"]

            if loo_mean.size:
                ax.plot(x, loo_mean, color="#1f77b4", lw=2.0, label="LOO order")
                if loo_lo.size:
                    ax.fill_between(x, loo_lo, loo_hi, color="#1f77b4", alpha=0.15)
            if rnd_mean.size:
                ax.plot(x, rnd_mean, color="#2ca02c", lw=1.6, ls="--", label="Random")
                if rnd_lo.size:
                    ax.fill_between(x, rnd_lo, rnd_hi, color="#2ca02c", alpha=0.12)

            if c == 0:
                ax.set_ylabel("Cumulative ΔM (higher is better)")
            ax.set_xlabel("Top-k spans removed")
            ax.set_ylim(y_min, y_max)

            rep = stats.get("rep_bundle")
            if rep is not None and loo_mean.size:
                rep_curve = cumulative_curve_in_rank_order(rep)
                rep_curve = rep_curve[: len(x)]
                annotate_steepest(ax, rep, rep_curve)

        handles = [
            plt.Line2D([0], [0], color="#1f77b4", lw=2.0, label="LOO order"),
            plt.Line2D([0], [0], color="#2ca02c", lw=1.6, ls="--", label="Random"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"{title} – {case_label}", y=0.995, fontsize=16)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))

        png_path = output_dir / f"loo_deletion_curves_{case_key}.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

    return output_dir / "loo_deletion_curves.png"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Deletion Faithfulness (LOO) curves from results JSONs.")
    parser.add_argument("--model", type=str, default="meta-llama_Llama-3.2-1B-Instruct", help="Model subdirectory under results/ to read.")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Root results directory (default: results)")
    parser.add_argument("--output-dir", type=Path, default=Path("fig"), help="Where to write the figure (default: fig)")
    parser.add_argument("--n-random", type=int, default=200, help="Random baseline samples per run (default: 200)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args(argv)

    results_dir = (args.results_root / args.model).resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    groups = find_runs(results_dir)
    out = plot_grid(groups, args.output_dir, title="Deletion Faithfulness (LOO)", n_random=args.n_random, seed=args.seed)
    print(f"[ok] Wrote {out}")


if __name__ == "__main__":
    main()
