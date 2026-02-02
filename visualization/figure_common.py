#!/usr/bin/env python3
"""
Shared utilities for figure-oriented scripts.

Provides configuration for benchmark suites alongside helpers for loading,
transforming, and selecting span-level metrics used by the plotting scripts.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "SuiteConfig",
    "SUITES",
    "SUITE_PALETTE",
    "SUITE_LABEL_PALETTE",
    "PROTECTIVE_COLOR",
    "ensure_exists",
    "prepare_suite_payload",
]


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

RUN_PRIORITY: Tuple[str, ...] = ("_final_", "_fixed_", "_run")
SUITE_PALETTE = {suite.key: suite.color for suite in SUITES}
SUITE_LABEL_PALETTE = {suite.label: suite.color for suite in SUITES}
PROTECTIVE_COLOR = "#1b9e77"


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {kind} at {path} but it was not found.")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _select_primary_run(run_ids: Sequence[str]) -> Optional[str]:
    ordered = sorted(run_ids)
    if not ordered:
        return None
    for token in RUN_PRIORITY:
        candidates = [rid for rid in ordered if token in rid]
        if candidates:
            return sorted(candidates)[-1]
    return ordered[-1]


def _load_run_payload(results_dir: Path, run_id: str) -> Optional[Dict[str, object]]:
    run_path = results_dir / f"{run_id}.json"
    if not run_path.exists():
        return None
    try:
        with run_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _extract_base_scores(payload: Dict[str, object]) -> Optional[Dict[str, float]]:
    base_scores = payload.get("base_scores")
    if not isinstance(base_scores, dict):
        return None
    try:
        return {
            "m_full": float(base_scores.get("m(C)")),
            "m_minimal": float(base_scores.get("m(M)")),
        }
    except (TypeError, ValueError):
        return None


def _extract_intent_aoi(payload: Dict[str, object]) -> pd.DataFrame:
    intent_map = payload.get("intent_aoi")
    records: List[Dict[str, object]] = []
    if isinstance(intent_map, dict):
        for intent_text, values in intent_map.items():
            if not isinstance(values, dict):
                continue
            delta = values.get("ΔM(M)")
            if delta is None:
                continue
            try:
                delta_val = float(delta)
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "intent_raw": intent_text,
                    "span_label": _format_span_label(intent_text),
                    "aoi_minimal": delta_val,
                }
            )
    df_intents = pd.DataFrame.from_records(records)
    if not df_intents.empty:
        df_intents["abs_aoi_minimal"] = df_intents["aoi_minimal"].abs()
    return df_intents


def _format_span_label(raw: str) -> str:
    if not raw:
        return ""
    text = str(raw).strip().replace("\n", " ")
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    text = textwrap.shorten(text, width=110, placeholder="...")
    return textwrap.fill(text, width=34)


def _describe_distribution(values: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return None
    median = float(np.median(finite_vals))
    p25 = float(np.percentile(finite_vals, 25))
    p75 = float(np.percentile(finite_vals, 75))
    return median, p25, p75


def prepare_suite_payload(
    span_metrics: Path,
    *,
    model_name: str,
    top_k: int,
    include_base_scores: bool = False,
    include_intent_aoi: bool = False,
    results_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Load span metrics for the requested model and compute per-suite payloads.
    """
    ensure_exists(span_metrics, "span metrics CSV")
    df = pd.read_csv(span_metrics)
    if "model_name" not in df.columns:
        raise ValueError("span metrics CSV must include a 'model_name' column.")
    df = df[df["model_name"] == model_name].copy()
    if df.empty:
        raise ValueError(f"No rows found for model '{model_name}'.")

    for col in ("aoi_empty", "aoi_minimal", "aoi_leave_one_out", "loo"):
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    if (include_base_scores or include_intent_aoi) and results_dir is None:
        raise ValueError(
            "results_dir must be provided when base scores or intent AOI data are requested."
        )
    if results_dir is not None:
        ensure_exists(results_dir, "results directory")

    suite_payload: Dict[str, Dict[str, object]] = {}
    for suite in SUITES:
        mask = df["case_id"].astype(str).str.startswith(suite.case_prefix)
        df_suite = df[mask].copy()
        if df_suite.empty:
            continue

        run_id = _select_primary_run(df_suite["run_id"].unique())
        if run_id:
            df_suite = df_suite[df_suite["run_id"] == run_id]

        df_suite = df_suite.assign(
            span_label=lambda frame: frame["span_text"].apply(_format_span_label)
        )
        df_suite["abs_loo"] = df_suite["loo"].abs()
        top_spans = (
            df_suite.dropna(subset=["loo"])
            .sort_values("abs_loo", ascending=False)
            .head(top_k)
        )

        aoi_vals = df_suite["aoi_minimal"].dropna().to_numpy(dtype=float)
        aoi_stats = _describe_distribution(aoi_vals)

        base_scores: List[Dict[str, object]] = []
        intent_df: Optional[pd.DataFrame] = None
        run_payload: Optional[Dict[str, object]] = None
        if (include_base_scores or include_intent_aoi) and run_id and results_dir is not None:
            run_payload = _load_run_payload(results_dir, run_id)

        if include_base_scores and run_payload:
            info = _extract_base_scores(run_payload)
            if info:
                base_scores.append({"run_id": run_id, **info})

        if include_intent_aoi and run_payload:
            df_intent = _extract_intent_aoi(run_payload)
            if not df_intent.empty:
                intent_df = df_intent

        suite_payload[suite.key] = {
            "config": suite,
            "run_id": run_id,
            "top_spans": top_spans,
            "aoi_values": aoi_vals,
            "aoi_stats": aoi_stats,
            "base_scores": base_scores,
            "intent_aoi": intent_df,
            "suite_df": df_suite.copy(),
        }

    if not suite_payload:
        raise ValueError("No suite data could be extracted from span metrics.")

    return suite_payload
