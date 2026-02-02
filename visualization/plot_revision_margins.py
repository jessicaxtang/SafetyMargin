#!/usr/bin/env python3
"""
Plot base-score shifts (``f(C)`` or ``m(C)``) before vs after suite revisions.

Usage example:
  python visualization/plot_revision_margins.py \
      --before-dir results/meta-llama_Llama-3.2-1B-Instruct \
      --after-dir results/meta-llama_Llama-3.2-1B-Instruct \
      --output fig/analysis/revision_margins.png \
      --metric margin
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)


@dataclass(frozen=True)
class SuiteConfig:
    key: str
    label: str
    case_prefix: str
    color: str


SUITES: Tuple[SuiteConfig, ...] = (
    SuiteConfig("privacy", "Privacy", "PRIVACY_", "#1f77b4"),
    SuiteConfig("injection", "Injection", "INJECTION_", "#d62728"),
    SuiteConfig("evidence", "Evidence", "EVIDENCE_", "#2ca02c"),
    SuiteConfig("role_confusion", "Role Confusion", "ROLE_", "#9467bd"),
    SuiteConfig("safety", "Safety", "SAFETY_", "#ff7f0e"),
)

SUITE_LABEL_PALETTE: Dict[str, str] = {suite.label: suite.color for suite in SUITES}
UNKNOWN_SUITE_COLOR = "#6e6e6e"


@dataclass(frozen=True)
class RunRecord:
    case_id: str
    case_title: str
    run_id: str
    suite_key: str
    suite_label: str
    score: float
    minimal_score: float
    path: Path
    phase: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-dir",
        type=Path,
        required=True,
        help="Directory containing pre-revision run JSONs (e.g. *_run*.json).",
    )
    parser.add_argument(
        "--after-dir",
        type=Path,
        required=True,
        help="Directory containing post-revision run JSONs (e.g. *_fixed_run*.json).",
    )
    parser.add_argument(
        "--before-pattern",
        default="*_run*.json",
        help="Glob pattern to select pre-revision artefacts inside --before-dir.",
    )
    parser.add_argument(
        "--after-pattern",
        default="*_fixed_run*.json",
        help="Glob pattern to select post-revision artefacts inside --after-dir.",
    )
    parser.add_argument(
        "--before-label",
        default="Before revision",
        help="Axis label for the pre-revision runs.",
    )
    parser.add_argument(
        "--after-label",
        default="After revision",
        help="Axis label for the post-revision runs.",
    )
    parser.add_argument(
        "--metric",
        choices=("probability", "margin"),
        default="margin",
        help="Which base score to plot: probability f(C) or directional margin m(C).",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Optional case_id allowlist; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fig/analysis/revision_margins.png"),
        help="Destination for the rendered plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure save DPI.",
    )
    return parser.parse_args()


def resolve_suite(case_id: Optional[str]) -> Tuple[str, str]:
    if not case_id:
        return "unknown", "Unknown"
    prefix = case_id.split("_", 1)[0].lower()
    for suite in SUITES:
        if suite.key == prefix:
            return suite.key, suite.label
    upper_case = case_id.upper()
    for suite in SUITES:
        if upper_case.startswith(suite.case_prefix.upper()):
            return suite.key, suite.label
    label = prefix.replace("_", " ").title()
    return prefix, label


def load_metric_records(paths: Iterable[Path], phase: str, metric: str) -> List[RunRecord]:
    score_key = "f(C)" if metric == "probability" else "m(C)"
    minimal_key = "f(M)" if metric == "probability" else "m(M)"
    records: List[RunRecord] = []
    for json_path in sorted(paths):
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f"[warn] Skipping {json_path}: {exc}")
            continue
        base_scores = payload.get("base_scores") or {}
        score = base_scores.get(score_key)
        minimal = base_scores.get(minimal_key)
        if score is None or minimal is None:
            continue
        try:
            score_val = float(score)
            minimal_val = float(minimal)
        except (TypeError, ValueError):
            continue
        meta = payload.get("meta", {})
        case_id = meta.get("case_id") or json_path.stem
        case_title = meta.get("case_title") or case_id
        suite_key, suite_label = resolve_suite(case_id)
        records.append(
            RunRecord(
                case_id=case_id,
                case_title=case_title,
                run_id=json_path.stem,
                suite_key=suite_key,
                suite_label=suite_label,
                score=score_val,
                minimal_score=minimal_val,
                path=json_path,
                phase=phase,
            )
        )
    return records


def pair_records(
    before: List[RunRecord],
    after: List[RunRecord],
    case_allowlist: Optional[List[str]],
) -> Dict[str, Dict[str, RunRecord]]:
    pairs: Dict[str, Dict[str, RunRecord]] = {}
    allow = set(case_allowlist) if case_allowlist else None

    for record in before:
        if allow and record.case_id not in allow:
            continue
        pairs.setdefault(record.case_id, {})["before"] = record

    for record in after:
        if allow and record.case_id not in allow:
            continue
        pairs.setdefault(record.case_id, {})["after"] = record

    return {cid: data for cid, data in pairs.items() if "before" in data and "after" in data}


def plot_pairs(
    pairs: Dict[str, Dict[str, RunRecord]],
    before_label: str,
    after_label: str,
    metric: str,
    output_path: Path,
    dpi: int,
) -> None:
    if not pairs:
        raise ValueError("No paired cases found. Check your directories/patterns.")

    ordered_pairs = sorted(
        pairs.items(),
        key=lambda item: (item[1]["before"].suite_key, item[0]),
    )

    suites_to_pairs: Dict[str, List[Tuple[str, Dict[str, RunRecord]]]] = defaultdict(list)
    for case_id, payload in ordered_pairs:
        label = payload["before"].suite_label
        suites_to_pairs[label].append((case_id, payload))

    ordered_suite_labels: List[str] = [
        suite.label for suite in SUITES if suite.label in suites_to_pairs
    ]
    extra_labels = [label for label in suites_to_pairs if label not in ordered_suite_labels]
    ordered_suite_labels.extend(sorted(extra_labels))

    if not ordered_suite_labels:
        raise ValueError("No suites found within paired records.")

    suite_index = {label: idx for idx, label in enumerate(ordered_suite_labels)}

    rows: List[Dict[str, object]] = []
    deltas: List[float] = []
    suite_deltas: Dict[str, List[float]] = defaultdict(list)
    suite_after_max: Dict[str, float] = defaultdict(lambda: float("-inf"))
    suite_minimal_values: Dict[str, List[float]] = defaultdict(list)

    for label in ordered_suite_labels:
        idx = suite_index[label]
        for case_id, payload in suites_to_pairs[label]:
            before = payload["before"]
            after = payload["after"]
            minimal_val = before.minimal_score
            if np.isnan(minimal_val) and not np.isnan(after.minimal_score):
                minimal_val = after.minimal_score
            pair_id = f"{before.suite_key}_{case_id}"
            if not np.isnan(minimal_val):
                suite_minimal_values[label].append(minimal_val)
            rows.append(
                {
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "suite_label": label,
                    "phase": before_label,
                    "x": idx,
                    "score": before.score,
                }
            )
            rows.append(
                {
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "suite_label": label,
                    "phase": after_label,
                    "x": idx + 0.3,
                    "score": after.score,
                }
            )
            delta_val = after.score - before.score
            deltas.append(delta_val)
            suite_deltas[label].append(delta_val)
            top_val = max(after.score, before.score)
            suite_after_max[label] = max(suite_after_max[label], top_val)

    df = pd.DataFrame(rows)
    palette = {
        label: SUITE_LABEL_PALETTE.get(label, UNKNOWN_SUITE_COLOR)
        for label in ordered_suite_labels
    }

    fig_width = max(6.5, 1.6 * len(ordered_suite_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.4))

    sns.lineplot(
        data=df,
        x="x",
        y="score",
        hue="suite_label",
        units="pair_id",
        estimator=None,
        marker="o",
        linewidth=2,
        palette=palette,
        ax=ax,
    )

    if metric == "probability":
        ylabel = r"Pass probability $f(C)$"
        title = "Pass probabilities before vs after revision"
        ref_value = 0.5
        delta_label = "Δf(C)"
    else:
        ylabel = r"Directional margin $m(C)$"
        title = "Directional margins before vs after revision"
        ref_value = 0.0
        delta_label = "Δm(C)"

    ax.set_xlim(-0.6, len(ordered_suite_labels) - 0.4)
    ax.set_xticks([suite_index[label] for label in ordered_suite_labels])
    ax.set_xticklabels(ordered_suite_labels)
    ax.set_xlabel("Suite")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.axhline(ref_value, color="#888888", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=9)

    legend = ax.legend(title="Suite", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    if legend:
        legend_title = legend.get_title()
        if legend_title:
            legend_title.set_fontsize(12)
        for text in legend.get_texts():
            text.set_fontsize(10)

    baseline_stats = {
        label: float(np.median(values)) for label, values in suite_minimal_values.items() if values
    }
    all_scores = df["score"].dropna().tolist()
    if baseline_stats:
        all_scores.extend(baseline_stats.values())
    y_min = float(min(all_scores)) if all_scores else 0.0
    y_max = float(max(all_scores)) if all_scores else 1.0
    y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.2
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for label, baseline in baseline_stats.items():
        idx = suite_index[label]
        color = palette.get(label, UNKNOWN_SUITE_COLOR)
        ax.hlines(
            baseline,
            idx - 0.32,
            idx + 0.32,
            colors=color,
            linestyles=(0, (4, 2)),
            linewidth=2.0,
            alpha=0.9,
        )
        # Place label slightly left of the baseline segment to avoid overlap with lines/markers
        ax.text(
            idx - 0.36,
            baseline,
            "Minimal",
            fontsize=9,
            color=color,
            ha="right",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "none", "alpha": 0.8},
            clip_on=False,
        )

    for label in ordered_suite_labels:
        deltas_for_suite = suite_deltas.get(label)
        max_after = suite_after_max.get(label)
        if not deltas_for_suite or max_after == float("-inf"):
            continue
        median_delta_suite = float(np.median(deltas_for_suite))
        idx = suite_index[label]
        # Stagger label heights by suite index to reduce cross-suite collisions
        offset = 0.22 if (idx % 2) else 0.32
        ax.text(
            idx,
            max_after + y_pad * offset,
            f"{delta_label}̃={median_delta_suite:+.3f}",
            fontsize=10,
            color=palette.get(label, UNKNOWN_SUITE_COLOR),
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "none", "alpha": 0.85},
            clip_on=False,
        )

    median_delta = float(np.median(deltas))
    # ax.text(
    #     0.0,
    #     -0.16,
    #     f"Median {delta_label} (after-before): {median_delta:+.3f}",
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     color="#333333",
    # )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote revision comparison to {output_path}")


def main() -> None:
    args = parse_args()

    before_paths = [
        path
        for path in sorted(args.before_dir.glob(args.before_pattern))
        if all(token not in path.stem for token in ("_fixed_", "_repair_", "_final_"))
    ]
    after_paths = [
        path
        for path in sorted(args.after_dir.glob(args.after_pattern))
        if "_fixed_" in path.stem
    ]

    if not before_paths:
        raise FileNotFoundError(f"No files matched {args.before_pattern} in {args.before_dir}")
    if not after_paths:
        raise FileNotFoundError(f"No files matched {args.after_pattern} in {args.after_dir}")

    before_records = load_metric_records(before_paths, phase=args.before_label, metric=args.metric)
    after_records = load_metric_records(after_paths, phase=args.after_label, metric=args.metric)

    pairs = pair_records(before_records, after_records, case_allowlist=args.case_id)
    plot_pairs(
        pairs,
        before_label=args.before_label,
        after_label=args.after_label,
        metric=args.metric,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
