#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate reference attribution outputs across runs into run-level and grouped CSV summaries."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[Path("experiments"), Path("experiments-handcrafted-promptloo-helpful")],
        help="Root directories to scan for reference_attribution_n*_seed*/summary.json",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/reference_attribution_n*_seed*/summary.json",
        help="Glob pattern used under each root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/out/reference_attribution_eval"),
        help="Directory to write CSV outputs.",
    )
    parser.add_argument(
        "--abs-attribution-threshold",
        type=float,
        default=0.2,
        help="Threshold for counting high-magnitude per-unit attributions.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _iter_summary_paths(roots: Iterable[Path], pattern: str) -> List[Path]:
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(sorted(root.glob(pattern)))
    unique = sorted(set(paths))
    return unique


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_per_unit_metrics(per_unit_path: Path, threshold: float) -> Dict[str, float]:
    if not per_unit_path.exists():
        return {
            "n_unit_rows_csv": 0.0,
            "helps_rate_csv": float("nan"),
            "hurts_rate_csv": float("nan"),
            "neutral_rate_csv": float("nan"),
            "high_abs_rate_csv": float("nan"),
        }

    helps = 0
    hurts = 0
    neutral = 0
    high_abs = 0
    total = 0

    with per_unit_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            label = str(row.get("contribution_label", "")).strip().lower()
            if label == "helps_margin":
                helps += 1
            elif label == "hurts_margin":
                hurts += 1
            else:
                neutral += 1

            attribution = _safe_float(row.get("attribution", float("nan")))
            if not math.isnan(attribution) and abs(attribution) >= threshold:
                high_abs += 1

    if total == 0:
        return {
            "n_unit_rows_csv": 0.0,
            "helps_rate_csv": float("nan"),
            "hurts_rate_csv": float("nan"),
            "neutral_rate_csv": float("nan"),
            "high_abs_rate_csv": float("nan"),
        }

    return {
        "n_unit_rows_csv": float(total),
        "helps_rate_csv": float(helps) / float(total),
        "hurts_rate_csv": float(hurts) / float(total),
        "neutral_rate_csv": float(neutral) / float(total),
        "high_abs_rate_csv": float(high_abs) / float(total),
    }


def _mean(values: List[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _std(values: List[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan")
    mu = sum(vals) / n
    return float((sum((x - mu) ** 2 for x in vals) / n) ** 0.5)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    summary_paths = _iter_summary_paths(args.roots, args.glob)
    run_rows: List[Dict[str, Any]] = []

    for summary_path in summary_paths:
        try:
            payload = _load_json(summary_path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        cli_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
        run_dir = summary_path.parent
        per_unit_path = run_dir / "per_unit_rows.csv"
        per_unit_metrics = _load_per_unit_metrics(per_unit_path, threshold=float(args.abs_attribution_threshold))

        row: Dict[str, Any] = {
            "summary_path": str(summary_path),
            "run_dir": str(run_dir),
            "dataset": str(cli_args.get("dataset", "unknown")),
            "local_dataset": str(cli_args.get("local_dataset", "")),
            "intervention_mode": str(cli_args.get("intervention_mode", "unknown")),
            "prompt_unit_splitter": str(cli_args.get("prompt_unit_splitter", "")),
            "margin_metric": str(payload.get("margin_metric", cli_args.get("margin_metric", "unknown"))),
            "base_model": str(cli_args.get("base_model", "unknown")),
            "seed": cli_args.get("seed", None),
            "n_requested": _safe_float(payload.get("n_requested", float("nan"))),
            "n_loaded": _safe_float(payload.get("n_loaded", float("nan"))),
            "n_examples_evaluated": _safe_float(payload.get("n_examples_evaluated", float("nan"))),
            "n_examples_skipped": _safe_float(payload.get("n_examples_skipped", float("nan"))),
            "n_unit_rows": _safe_float(payload.get("n_unit_rows", float("nan"))),
            "chosen_beats_rejected_rate": _safe_float(payload.get("chosen_beats_rejected_rate", float("nan"))),
            "full_margin_mean": _safe_float(payload.get("full_margin_mean", float("nan"))),
            "full_margin_std": _safe_float(payload.get("full_margin_std", float("nan"))),
            "attribution_abs_mean": _safe_float(payload.get("attribution_abs_mean", float("nan"))),
            "attribution_helps_rate": _safe_float(payload.get("attribution_helps_rate", float("nan"))),
            "validation_match_rate": _safe_float(payload.get("validation_match_rate", float("nan"))),
        }
        row.update(per_unit_metrics)
        run_rows.append(row)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_runs_path = out_dir / "all_reference_attribution_runs.csv"
    _write_csv(all_runs_path, run_rows)

    grouped: Dict[Tuple[str, str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in run_rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("intervention_mode", "")),
            str(row.get("margin_metric", "")),
            str(row.get("base_model", "")),
        )
        for metric in [
            "chosen_beats_rejected_rate",
            "full_margin_mean",
            "attribution_abs_mean",
            "attribution_helps_rate",
            "validation_match_rate",
            "helps_rate_csv",
            "hurts_rate_csv",
            "high_abs_rate_csv",
        ]:
            grouped[key][metric].append(_safe_float(row.get(metric, float("nan"))))

    grouped_rows: List[Dict[str, Any]] = []
    for (dataset, intervention_mode, margin_metric, base_model), metric_dict in sorted(grouped.items()):
        grouped_row: Dict[str, Any] = {
            "dataset": dataset,
            "intervention_mode": intervention_mode,
            "margin_metric": margin_metric,
            "base_model": base_model,
            "n_runs": len(metric_dict.get("chosen_beats_rejected_rate", [])),
        }
        for metric, values in metric_dict.items():
            grouped_row[f"{metric}_mean"] = _mean(values)
            grouped_row[f"{metric}_std"] = _std(values)
        grouped_rows.append(grouped_row)

    grouped_path = out_dir / "grouped_reference_attribution_summary.csv"
    _write_csv(grouped_path, grouped_rows)

    print(f"Found runs: {len(run_rows)}")
    print(f"All-runs CSV: {all_runs_path}")
    print(f"Grouped CSV: {grouped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
