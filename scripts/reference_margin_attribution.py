#!/usr/bin/env python3
"""Evaluate reference-margin attribution on HH-RLHF chosen/rejected pairs.

This script answers:
- Does the teacher-forced preference margin separate dataset references (chosen > rejected)?
- Under leave-one-out prompt interventions, which units/rules contribute positively or negatively
  to that reference margin?

Outputs:
- summary.json: run-level metrics
- examples.csv: per-example full-margin diagnostics
- per_unit_rows.csv: per-example, per-LOO-unit attribution rows
- unit_aggregate.csv: aggregated attribution by unit index/text
"""

from __future__ import annotations

import argparse
import csv
import sys
import json
import re
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional, Set

import numpy as np
from tqdm.auto import tqdm

import nltk
nltk.download("punkt", quiet=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetymargin.models.huggingface_wrapper import HFModel
from safetymargin.preference_transfer import (
    Example,
    DEFAULT_POLICY_RULES,
    build_leave_one_out_prompt_variants,
    load_hh_rlhf,
    set_seed,
)

try:
    from nltk.tokenize import sent_tokenize
except Exception:
    sent_tokenize = None  # type: ignore[assignment]


def _is_cuda_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _maybe_clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _extract_score_from_teacher_forcing_result(result: Dict[str, Any], normalize_by_length: bool) -> float:
    if not normalize_by_length:
        return float(result["sum_log_prob"])

    if "mean_log_prob" in result and result["mean_log_prob"] is not None:
        return float(result["mean_log_prob"])

    target_log_probs = result.get("target_log_probs")
    if target_log_probs is None:
        return float(result["sum_log_prob"])

    try:
        token_count = max(int(target_log_probs.numel()), 1)
    except Exception:
        token_count = max(len(target_log_probs), 1)
    return float(result["sum_log_prob"]) / float(token_count)


def _standardize_per_example(
    example_indices: Sequence[int],
    values: Sequence[float],
    min_count: int = 2,
) -> Tuple[List[float], int]:
    """Return z-scored values per example index (mean 0 / std 1 within each example)."""
    if len(example_indices) != len(values):
        raise ValueError("example_indices and values must have the same length")

    z_scores = [float("nan")] * len(values)
    grouped: Dict[int, List[int]] = {}
    for row_idx, example_index in enumerate(example_indices):
        grouped.setdefault(int(example_index), []).append(row_idx)

    standardized_examples = 0
    for indices in grouped.values():
        valid_indices = [
            idx for idx in indices if not math.isnan(float(values[idx]))
        ]
        if len(valid_indices) < min_count:
            continue
        arr = np.asarray([float(values[idx]) for idx in valid_indices], dtype=np.float64)
        std = float(np.std(arr))
        if std <= 1e-12:
            continue
        mean = float(np.mean(arr))
        for pos, idx in enumerate(valid_indices):
            z_scores[idx] = float((arr[pos] - mean) / std)
        standardized_examples += 1

    return z_scores, standardized_examples


def _teacher_forced_score(model: Any, prompt: str, target: str, normalize_by_length: bool) -> float:
    result = model.teacher_forcing_forward(prompt_text=prompt, target_text=target, require_grad=False)
    return _extract_score_from_teacher_forcing_result(result, normalize_by_length=normalize_by_length)


def _compute_reference_margins_batched(
    model: Any,
    prompts: Sequence[str],
    chosen: str,
    rejected: str,
    batch_size: int,
    normalize_by_length: bool,
) -> List[float]:
    if len(prompts) == 0:
        return []

    if not hasattr(model, "teacher_forcing_forward_batch"):
        margins: List[float] = []
        for prompt in prompts:
            chosen_score = _teacher_forced_score(model, prompt, chosen, normalize_by_length=normalize_by_length)
            rejected_score = _teacher_forced_score(model, prompt, rejected, normalize_by_length=normalize_by_length)
            margins.append(chosen_score - rejected_score)
        return margins

    margins: List[float] = []
    step = max(int(batch_size), 1)
    start = 0
    while start < len(prompts):
        chunk = list(prompts[start : start + step])
        try:
            chosen_results = model.teacher_forcing_forward_batch(chunk, [chosen] * len(chunk), require_grad=False)
            rejected_results = model.teacher_forcing_forward_batch(chunk, [rejected] * len(chunk), require_grad=False)
        except Exception as exc:
            if _is_cuda_oom(exc):
                _maybe_clear_cuda_cache()
                if step > 1:
                    step = max(step // 2, 1)
                    continue
                # step == 1: fallback to non-batched scorer for this prompt
                prompt = chunk[0]
                chosen_score = _teacher_forced_score(model, prompt, chosen, normalize_by_length=normalize_by_length)
                rejected_score = _teacher_forced_score(model, prompt, rejected, normalize_by_length=normalize_by_length)
                margins.append(chosen_score - rejected_score)
                start += 1
                continue
            raise

        for chosen_result, rejected_result in zip(chosen_results, rejected_results):
            chosen_score = _extract_score_from_teacher_forcing_result(chosen_result, normalize_by_length=normalize_by_length)
            rejected_score = _extract_score_from_teacher_forcing_result(rejected_result, normalize_by_length=normalize_by_length)
            margins.append(chosen_score - rejected_score)
        start += len(chunk)
    return margins


def _split_prompt_units_nltk(prompt: str) -> List[str]:
    text = (prompt or "").strip()
    if not text:
        return []

    if sent_tokenize is None:
        raise RuntimeError(
            "NLTK is not installed. Install it with `pip install nltk` to use --prompt-unit-splitter nltk_sentence."
        )

    try:
        units = [chunk.strip() for chunk in sent_tokenize(text) if chunk and chunk.strip()]
    except LookupError:
        try:
            import nltk

            # Newer NLTK uses punkt_tab; keep punkt for compatibility.
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            units = [chunk.strip() for chunk in sent_tokenize(text) if chunk and chunk.strip()]
        except Exception:
            units = []

    if len(units) >= 2:
        return units

    fallback = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk and chunk.strip()]
    return fallback if len(fallback) >= 2 else []


def _split_local_prompt_units(prompt: str, splitter: str) -> List[str]:
    full_prompt = (prompt or "").strip()
    if not full_prompt:
        return []

    splitter_key = (splitter or "default").strip().lower()
    if splitter_key == "nltk_sentence":
        units = _split_prompt_units_nltk(full_prompt)
    else:
        units = [chunk.strip() for chunk in re.split(r"\n{2,}", full_prompt) if chunk and chunk.strip()]
        if len(units) < 2:
            units = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", full_prompt) if chunk and chunk.strip()]
    return units


def _join_prompt_units(units: Sequence[str], keep_indices: Set[int]) -> str:
    if not units or not keep_indices:
        return ""
    kept = [unit for idx, unit in enumerate(units) if idx in keep_indices]
    return " ".join(kept).strip()


def _find_assistant_unit_indices(units: Sequence[str]) -> Set[int]:
    indices: Set[int] = set()
    for idx, unit in enumerate(units):
        if "assistant:" in unit.lower():
            indices.add(idx)
    return indices


def _load_prompt_unit_metadata(path: Path) -> Dict[int, Dict[int, Dict[str, str]]]:
    metadata: Dict[int, Dict[int, Dict[str, str]]] = defaultdict(dict)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt-unit metadata file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file {path} is missing a header row")
        required = {"example_index", "unit_index", "unit_type"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Metadata file {path} missing required columns: {sorted(missing)}"
            )

        for row in reader:
            try:
                example_idx = int(row["example_index"])
                unit_idx = int(row["unit_index"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid example/unit index in {path}: {row}"
                ) from exc

            normalized: Dict[str, str] = {}
            for key, value in row.items():
                normalized[key] = value.strip() if isinstance(value, str) else value
            unit_type = (normalized.get("unit_type") or "").lower()
            normalized["unit_type"] = unit_type
            metadata.setdefault(example_idx, {})[unit_idx] = normalized

    return metadata


def _build_add_one_in_plan(
    units: Sequence[str],
    metadata: Dict[int, Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not units:
        return None, "no_units"

    user_indices = sorted(
        idx for idx, meta in metadata.items() if (meta.get("unit_type") or "").lower() == "user"
    )
    spec_indices = sorted(
        idx for idx, meta in metadata.items() if (meta.get("unit_type") or "").lower() == "spec"
    )
    if not user_indices:
        return None, "no_user_units"
    if not spec_indices:
        return None, "no_spec_units"

    assistant_indices = _find_assistant_unit_indices(units)
    base_keep: Set[int] = set(user_indices) | assistant_indices
    base_prompt = _join_prompt_units(units, base_keep)
    if not base_prompt or "Assistant:" not in base_prompt:
        return None, "invalid_base_prompt"

    plan: Dict[str, Any] = {
        "base_prompt": base_prompt,
        "variants": [],
        "assistant_indices": assistant_indices,
    }
    seen_prompts: Set[str] = {base_prompt}

    def _add_variant(idx: int, keep: Set[int], variant_type: str) -> None:
        prompt = _join_prompt_units(units, keep)
        if not prompt or "Assistant:" not in prompt:
            return
        if prompt in seen_prompts:
            return
        seen_prompts.add(prompt)
        plan["variants"].append(
            {
                "unit_index": idx,
                "unit_text": units[idx] if idx < len(units) else "",
                "prompt": prompt,
                "variant_type": variant_type,
            }
        )

    for spec_idx in spec_indices:
        keep = set(base_keep)
        keep.add(spec_idx)
        _add_variant(spec_idx, keep, variant_type="spec")

    for user_idx in user_indices:
        keep = set(assistant_indices)
        keep.add(user_idx)
        _add_variant(user_idx, keep, variant_type="user")

    if not plan["variants"]:
        return None, "no_valid_aoi_prompts"

    return plan, None


def _build_prompt_unit_variants_local(
    prompt: str,
    splitter: str,
) -> Tuple[str, List[Tuple[int, str, str]]]:
    full_prompt = (prompt or "").strip()
    if not full_prompt:
        return "", []

    units = _split_local_prompt_units(full_prompt, splitter)
    if len(units) < 2:
        return "", []

    variants: List[Tuple[int, str, str]] = []
    seen_reduced_prompts: set[str] = set()
    for unit_idx, unit_text in enumerate(units):
        kept = {idx for idx in range(len(units)) if idx != unit_idx}
        reduced_prompt = _join_prompt_units(units, kept)
        if not reduced_prompt or "Assistant:" not in reduced_prompt:
            continue
        if reduced_prompt in seen_reduced_prompts:
            continue
        seen_reduced_prompts.add(reduced_prompt)
        variants.append((unit_idx, unit_text, reduced_prompt))

    return full_prompt, variants


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _validate_local_row(row: Dict[str, Any], idx: int) -> Example:
    required = ["prompt", "chosen", "rejected"]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Local dataset row {idx} missing required fields: {missing}")

    prompt = str(row.get("prompt", "")).strip()
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()

    if not prompt:
        raise ValueError(f"Local dataset row {idx} has empty 'prompt'")
    if not chosen:
        raise ValueError(f"Local dataset row {idx} has empty 'chosen'")
    if not rejected:
        raise ValueError(f"Local dataset row {idx} has empty 'rejected'")

    return Example(prompt=prompt, chosen=chosen, rejected=rejected)


def load_local_dataset(path: str | Path, n_samples: int | None = None) -> List[Example]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset file not found: {path}")

    suffix = path.suffix.lower()
    rows: List[Dict[str, Any]] = []

    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_idx} in {path}: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ValueError(f"Line {line_idx} in {path} must be a JSON object")
                rows.append(parsed)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)
        if isinstance(parsed, dict):
            if "examples" in parsed and isinstance(parsed["examples"], list):
                rows = parsed["examples"]
            else:
                raise ValueError(
                    f"JSON object in {path} must contain an 'examples' list of {{prompt, chosen, rejected}} objects"
                )
        elif isinstance(parsed, list):
            rows = parsed
        else:
            raise ValueError(f"Unsupported JSON structure in {path}; expected list or object with 'examples'")
    else:
        raise ValueError(f"Unsupported local dataset extension '{suffix}'. Use .json, .jsonl, or .ndjson")

    if not rows:
        raise ValueError(f"No examples found in local dataset: {path}")

    dataset = [_validate_local_row(row, idx) for idx, row in enumerate(rows)]
    if n_samples is not None and n_samples > 0:
        dataset = dataset[:n_samples]
    return dataset


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference-margin attribution on HH-RLHF pairs")
    parser.add_argument(
        "--local-dataset",
        type=str,
        default=None,
        help="Optional local JSON/JSONL file with {prompt, chosen, rejected} objects. If set, skips HH-RLHF loading.",
    )
    parser.add_argument("--dataset", type=str, default="helpful", choices=["harmless", "helpful", "combined"])
    parser.add_argument("--hf-split", type=str, default="train", help="HF split for HH-RLHF")
    parser.add_argument("--n", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--max-length", type=int, default=None, help="Optional length filter for prompt/chosen/rejected")

    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct") # meta-llama/Meta-Llama-3-8B-Instruct
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default="auto")

    parser.add_argument(
        "--intervention-mode",
        type=str,
        default="prompt_units",
        choices=["prompt_units", "policy_rules"],
        help="Whether LOO ablates prompt units or behavior-policy rules",
    )
    parser.add_argument(
        "--prompt-unit-splitter",
        type=str,
        default="default",
        choices=["default", "nltk_sentence"],
        help="Splitter for prompt_units mode. Use nltk_sentence for sentence-level LOO on local handcrafted prompts.",
    )
    parser.add_argument(
        "--prompt-unit-labels",
        type=str,
        default=Path("dataset/handcrafted_v2/per_unit_rows_labelled2.csv"),
        help="Optional CSV with handcrafted prompt-unit metadata (columns: example_index, unit_index, unit_type, ...).",
    )
    parser.add_argument(
        "--add-one-in",
        # action="store_true",
        default=True,
        help="Compute add-one-in (AOI) margins for prompt units using --prompt-unit-labels metadata.",
    )
    parser.add_argument(
        "--policy-placement",
        type=str,
        default="prepend",
        choices=["prepend", "append"],
        help="Placement of behavior policy block relative to dataset prompt",
    )
    parser.add_argument(
        "--policy-rule",
        action="append",
        default=None,
        help="Behavior policy rule; repeat for multiple. Uses defaults if omitted.",
    )
    parser.add_argument("--reference-batch-size", type=int, default=8, help="Batch size for teacher-forced margin scoring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    margin_metric = "mean_log_prob"
    normalize_by_length = True

    prompt_unit_metadata: Dict[int, Dict[int, Dict[str, str]]] = {}
    if args.prompt_unit_labels:
        prompt_unit_metadata = _load_prompt_unit_metadata(Path(args.prompt_unit_labels))

    enable_add_one_in = bool(args.add_one_in)
    if enable_add_one_in and args.intervention_mode != "prompt_units":
        raise ValueError("--add-one-in is only supported for prompt_units intervention mode.")
    if enable_add_one_in and not prompt_unit_metadata:
        raise ValueError("--add-one-in requires --prompt-unit-labels metadata.")

    policy_rules = [rule.strip() for rule in (args.policy_rule or []) if rule and rule.strip()]
    if not policy_rules:
        policy_rules = list(DEFAULT_POLICY_RULES)

    if args.intervention_mode == "policy_rules" and len(policy_rules) < 2:
        raise ValueError("policy_rules intervention requires at least 2 non-empty rules")

    if args.local_dataset:
        dataset = load_local_dataset(Path(args.local_dataset), n_samples=args.n)
    else:
        dataset = load_hh_rlhf(
            split=args.dataset,
            n_samples=args.n,
            max_length=args.max_length,
            seed=args.seed,
            hf_split=args.hf_split,
        )

    model = HFModel(
        model_name=args.base_model,
        auto_load=True,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    example_rows: List[Dict[str, Any]] = []
    per_unit_rows: List[Dict[str, Any]] = []
    skipped_examples: List[Dict[str, Any]] = []
    aoi_skipped_examples: List[Dict[str, Any]] = []
    aoi_base_margins: List[float] = []
    aoi_deltas: List[float] = []

    iterator = tqdm(enumerate(dataset), total=len(dataset), desc="Reference attribution examples")
    for example_index, ex in iterator:
        example_metadata = prompt_unit_metadata.get(example_index)
        if args.local_dataset and args.intervention_mode == "prompt_units":
            full_prompt, variants = _build_prompt_unit_variants_local(
                prompt=ex.prompt,
                splitter=args.prompt_unit_splitter,
            )
        else:
            full_prompt, variants = build_leave_one_out_prompt_variants(
                prompt=ex.prompt,
                intervention_mode=args.intervention_mode,
                policy_rules=policy_rules,
                policy_placement=args.policy_placement,
            )

        if not variants:
            skipped_examples.append(
                {
                    "example_index": example_index,
                    "reason": "insufficient_units",
                    "prompt_chars": len(ex.prompt),
                }
            )
            continue

        aoi_unit_info: Dict[int, Dict[str, Any]] = {}
        aoi_base_margin: Optional[float] = None
        aoi_base_prompt: Optional[str] = None

        prompts = [full_prompt] + [variant[2] for variant in variants]
        margins = _compute_reference_margins_batched(
            model=model,
            prompts=prompts,
            chosen=ex.chosen,
            rejected=ex.rejected,
            batch_size=args.reference_batch_size,
            normalize_by_length=normalize_by_length,
        )

        full_margin = float(margins[0])
        full_chosen_score = _teacher_forced_score(model, full_prompt, ex.chosen, normalize_by_length=normalize_by_length)
        full_rejected_score = _teacher_forced_score(model, full_prompt, ex.rejected, normalize_by_length=normalize_by_length)

        example_row = {
            "example_index": example_index,
            "num_units": len(variants),
            "full_margin": full_margin,
            "full_chosen_logprob": full_chosen_score,
            "full_rejected_logprob": full_rejected_score,
            "margin_metric": margin_metric,
            "chosen_beats_rejected": int(full_margin > 0.0),
            "dataset_prompt": ex.prompt,
            "full_prompt": full_prompt,
        }
        example_rows.append(example_row)

        if enable_add_one_in:
            if example_metadata is None:
                aoi_skipped_examples.append(
                    {
                        "example_index": example_index,
                        "reason": "missing_metadata",
                    }
                )
            else:
                units_for_plan = _split_local_prompt_units(ex.prompt, args.prompt_unit_splitter)
                plan, plan_reason = _build_add_one_in_plan(units_for_plan, example_metadata)
                if plan is None:
                    aoi_skipped_examples.append(
                        {
                            "example_index": example_index,
                            "reason": plan_reason or "plan_failed",
                        }
                    )
                else:
                    variant_list = list(plan["variants"])
                    plan_prompts = [plan["base_prompt"]] + [variant["prompt"] for variant in variant_list]
                    plan_margins = _compute_reference_margins_batched(
                        model=model,
                        prompts=plan_prompts,
                        chosen=ex.chosen,
                        rejected=ex.rejected,
                        batch_size=args.reference_batch_size,
                        normalize_by_length=normalize_by_length,
                    )
                    if len(plan_margins) != len(plan_prompts):
                        aoi_skipped_examples.append(
                            {
                                "example_index": example_index,
                                "reason": "aoi_margin_mismatch",
                            }
                        )
                    else:
                        aoi_base_prompt = plan["base_prompt"]
                        aoi_base_margin = float(plan_margins[0])
                        aoi_base_margins.append(aoi_base_margin)
                        for variant, margin in zip(variant_list, plan_margins[1:]):
                            unit_idx = int(variant["unit_index"])
                            added_margin = float(margin)
                            delta_val = added_margin - aoi_base_margin
                            aoi_unit_info[unit_idx] = {
                                "aoi_margin": added_margin,
                                "aoi_delta": delta_val,
                                "aoi_prompt": variant["prompt"],
                                "aoi_variant_type": variant.get("variant_type", ""),
                            }
                            aoi_deltas.append(delta_val)
                        example_row["aoi_baseline_margin"] = aoi_base_margin
                        example_row["aoi_base_prompt"] = aoi_base_prompt
                        example_row["aoi_variant_count"] = len(aoi_unit_info)

        for variant_idx, (unit_index, unit_text, reduced_prompt) in enumerate(variants, start=1):
            reduced_margin = float(margins[variant_idx])
            delta_ref = reduced_margin - full_margin
            attribution = full_margin - reduced_margin
            contribution_label = "helps_margin" if attribution > 0 else ("hurts_margin" if attribution < 0 else "neutral")

            row_dict = {
                "example_index": example_index,
                "unit_index": unit_index,
                "num_units": len(variants),
                "unit_kind": "policy_rule" if args.intervention_mode == "policy_rules" else "prompt_unit",
                "unit_text": unit_text,
                "full_margin": full_margin,
                "reduced_margin": reduced_margin,
                "delta_ref": delta_ref,
                "attribution": attribution,
                "contribution_label": contribution_label,
                "margin_metric": margin_metric,
                "dataset_prompt": ex.prompt,
                "full_prompt": full_prompt,
                "reduced_prompt": reduced_prompt,
            }
            if example_metadata:
                meta = example_metadata.get(unit_index)
                if meta:
                    row_dict["unit_type"] = meta.get("unit_type", "")
                    if "ground_truth" in meta:
                        row_dict["ground_truth"] = meta.get("ground_truth", "")
            if enable_add_one_in:
                if aoi_base_margin is not None:
                    row_dict["aoi_baseline_margin"] = aoi_base_margin
                if unit_index in aoi_unit_info:
                    aoi_entry = aoi_unit_info[unit_index]
                    aoi_delta_val = float(aoi_entry["aoi_delta"])
                    row_dict["aoi_margin"] = float(aoi_entry["aoi_margin"])
                    row_dict["aoi_delta"] = aoi_delta_val
                    row_dict["aoi_contribution_label"] = (
                        "helps_margin"
                        if aoi_delta_val > 0
                        else ("hurts_margin" if aoi_delta_val < 0 else "neutral")
                    )
                    row_dict["aoi_prompt"] = aoi_entry["aoi_prompt"]
                    row_dict["aoi_variant_type"] = aoi_entry.get("aoi_variant_type", "")
                else:
                    row_dict["aoi_margin"] = None
                    row_dict["aoi_delta"] = None
                    row_dict["aoi_contribution_label"] = ""
                    row_dict["aoi_prompt"] = ""
                    row_dict["aoi_variant_type"] = ""

            per_unit_rows.append(row_dict)

    full_margins = [float(row["full_margin"]) for row in example_rows]
    positive_rate = float(np.mean([1.0 if m > 0 else 0.0 for m in full_margins])) if full_margins else float("nan")

    attributions = [float(row["attribution"]) for row in per_unit_rows]
    helps_rate = float(np.mean([1.0 if a > 0 else 0.0 for a in attributions])) if attributions else float("nan")

    grouped: Dict[Tuple[int, str], Dict[str, List[float]]] = defaultdict(lambda: {"loo": [], "aoi": []})
    for row in per_unit_rows:
        key = (int(row["unit_index"]), str(row["unit_text"]))
        grouped[key]["loo"].append(float(row["attribution"]))
        aoi_value = row.get("aoi_delta")
        if isinstance(aoi_value, (int, float)):
            grouped[key]["aoi"].append(float(aoi_value))

    unit_aggregate_rows: List[Dict[str, Any]] = []
    for (unit_index, unit_text), stats in sorted(grouped.items(), key=lambda item: item[0][0]):
        loo_arr = np.array(stats["loo"], dtype=float)
        row = {
            "unit_index": unit_index,
            "unit_text": unit_text,
            "count": int(loo_arr.size),
            "mean_attribution": float(np.mean(loo_arr)),
            "median_attribution": float(np.median(loo_arr)),
            "std_attribution": float(np.std(loo_arr)),
            "helps_rate": float(np.mean(loo_arr > 0)),
        }
        if stats["aoi"]:
            aoi_arr = np.array(stats["aoi"], dtype=float)
            row["mean_aoi_delta"] = float(np.mean(aoi_arr))
            row["median_aoi_delta"] = float(np.median(aoi_arr))
            row["std_aoi_delta"] = float(np.std(aoi_arr))
            row["aoi_helps_rate"] = float(np.mean(aoi_arr > 0))
        else:
            row["mean_aoi_delta"] = float("nan")
            row["median_aoi_delta"] = float("nan")
            row["std_aoi_delta"] = float("nan")
            row["aoi_helps_rate"] = float("nan")
        unit_aggregate_rows.append(row)

    delta_ref_values = [float(row.get("delta_ref", float("nan"))) for row in per_unit_rows]
    example_indices = [int(row.get("example_index", -1)) for row in per_unit_rows]
    delta_ref_z, n_z_examples = _standardize_per_example(example_indices, delta_ref_values)
    for row, z_value in zip(per_unit_rows, delta_ref_z):
        row["delta_ref_z"] = z_value

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "margin_metric": margin_metric,
        "policy_rules": policy_rules,
        "n_requested": int(args.n),
        "n_loaded": int(len(dataset)),
        "n_examples_evaluated": int(len(example_rows)),
        "n_examples_skipped": int(len(skipped_examples)),
        "n_unit_rows": int(len(per_unit_rows)),
        "full_margin_mean": float(np.mean(full_margins)) if full_margins else float("nan"),
        "full_margin_median": float(np.median(full_margins)) if full_margins else float("nan"),
        "full_margin_std": float(np.std(full_margins)) if full_margins else float("nan"),
        "chosen_beats_rejected_rate": positive_rate,
        "attribution_mean": float(np.mean(attributions)) if attributions else float("nan"),
        "attribution_median": float(np.median(attributions)) if attributions else float("nan"),
        "attribution_abs_mean": float(np.mean(np.abs(attributions))) if attributions else float("nan"),
        "attribution_helps_rate": helps_rate,
        "skipped_examples": skipped_examples,
        "delta_ref_z_examples": int(n_z_examples),
        "aoi_enabled": enable_add_one_in,
        "aoi_examples_with_baseline": int(len(aoi_base_margins)),
        "aoi_baseline_mean": float(np.mean(aoi_base_margins)) if aoi_base_margins else float("nan"),
        "aoi_baseline_std": float(np.std(aoi_base_margins)) if aoi_base_margins else float("nan"),
        "aoi_unit_rows": int(len(aoi_deltas)),
        "aoi_delta_mean": float(np.mean(aoi_deltas)) if aoi_deltas else float("nan"),
        "aoi_delta_median": float(np.median(aoi_deltas)) if aoi_deltas else float("nan"),
        "aoi_delta_std": float(np.std(aoi_deltas)) if aoi_deltas else float("nan"),
        "aoi_helps_rate": float(np.mean([1.0 if delta > 0 else 0.0 for delta in aoi_deltas])) if aoi_deltas else float("nan"),
        "aoi_skipped_examples": aoi_skipped_examples,
    }

    if args.base_model == "meta-llama/Llama-3.2-1B-Instruct":
        model_name = "llama-3.2-1b"
    elif args.base_model == "meta-llama/Meta-Llama-3-8B-Instruct":
        model_name = "llama-3-8b"
    elif args.base_model == "meta-llama/Llama-3.2-3B-Instruct":
        model_name = "llama-3.2-3b"
    elif args.base_model == "Qwen/Qwen2.5-1.5B-Instruct":
        model_name = "qwen-2.5-1.5b"
    elif args.base_model == "Qwen/Qwen2.5-3B-Instruct":
        model_name = "qwen-2.5-3b" 
    elif args.base_model == "Qwen/Qwen2.5-7B-Instruct":
        model_name = "qwen-2.5-7b"

    out_dir = Path(f"{args.output_dir}-{args.dataset}") / f"reference_attribution_n{args.n}_seed{args.seed}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    examples_path = out_dir / "examples.csv"
    per_unit_path = out_dir / "per_unit_rows.csv"
    unit_agg_path = out_dir / "unit_aggregate.csv"

    _write_csv(examples_path, example_rows)
    _write_csv(per_unit_path, per_unit_rows)
    _write_csv(unit_agg_path, unit_aggregate_rows)

    print("\nReference Attribution Summary")
    print(f"Examples evaluated: {summary['n_examples_evaluated']} / {summary['n_loaded']}")
    print(f"chosen_beats_rejected_rate: {summary['chosen_beats_rejected_rate']:.4f}")
    print(f"full_margin_mean: {summary['full_margin_mean']:.6f}")
    print(f"attribution_abs_mean: {summary['attribution_abs_mean']:.6f}")
    if enable_add_one_in:
        print(f"aoi_examples_with_baseline: {summary['aoi_examples_with_baseline']}")
        print(f"aoi_delta_mean: {summary['aoi_delta_mean']:.6f}")
        print(f"aoi_helps_rate: {summary['aoi_helps_rate']:.4f}")
    print(f"Summary JSON: {summary_path}")
    print(f"Examples CSV: {examples_path}")
    print(f"Per-unit CSV: {per_unit_path}")
    print(f"Unit aggregate CSV: {unit_agg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
