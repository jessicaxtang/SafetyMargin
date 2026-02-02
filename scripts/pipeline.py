#!/usr/bin/env python3
"""
SafetyMargin end-to-end attribution pipeline.

1. Load a case definition from JSON.
2. Materialize a concrete scenario (minimal context M, spans S, question Q).
3. Instantiate a model wrapper (HuggingFace).
4. Score the scenario with a log-probability-based norm scorer.
5. Compute LOO-style necessity signals and AOI-style sufficiency signals.
6. Print comprehensive report and optionally save JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from statistics import median
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # heavy dependency
    from safetymargin.models.huggingface_wrapper import HFModel
    HAS_HF_MODEL = True
    HF_MODEL_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # transformers/torch missing
    HFModel = None  # type: ignore
    HAS_HF_MODEL = False
    HF_MODEL_IMPORT_ERROR = exc

# Optional heavy imports used only when available
try:  # used only if HF model is present
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:  # allow stub mode
    HAS_TORCH = False

from safetymargin.datasets.scenarios import (
    Scenario,
    build_case_from_file,
    normalise_user_question,
)
from safetymargin.scorer import (
    LogProbNormConfig,
    LogProbNormScorer,
    DEFAULT_PASS_TEMPLATES,
    DEFAULT_FAIL_TEMPLATES,
)
from safetymargin.templates import resolve_template_list
from safetymargin.outputs import (
    generate_outputs,
    extract_generated_text,
)
from safetymargin.utils import (
    assemble_context,
    compute_base_scores,
    compute_loo,
    compute_aoi_baselines,
    compute_intent_aoi,
)
from safetymargin.reporting import print_report, format_section

try:
    from huggingface_hub import login as hf_login  # type: ignore
except Exception:
    hf_login = None  # type: ignore

token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if token and hf_login is not None:
    try:
        hf_login(token=token)
    except Exception:
        # Best-effort login; continue without cached credentials
        pass


def make_scalar(scorer: LogProbNormScorer, score_space: str):
    """Return a scalar attribution function f(ctx) per chosen score space.

    Convention: higher is safer (more refusal).
    - prob: use probability of refusal (safe class)
    - logit: normalized margin (lp_pass - lp_fail) / tau
    """
    space = (score_space or "prob").lower()
    if space == "prob":
        return scorer.f
    if space == "logit":
        return lambda ctx: (scorer.margin(ctx) / scorer.cfg.tau)
    raise ValueError(score_space)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        safe_dict = {}
        for key, val in value.items():
            if isinstance(key, (str, int, float, bool)) or key is None:
                safe_key = key
            else:
                safe_key = str(key)
            safe_dict[safe_key] = json_safe(val)
        return safe_dict
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def load_model(model_name: str, device: str, torch_dtype: Optional[str]) -> Any:
    """Instantiate the HuggingFace model wrapper."""
    if not HAS_HF_MODEL:
        raise RuntimeError(f"HFModel unavailable: {HF_MODEL_IMPORT_ERROR}")
    kwargs: Dict[str, Any] = {"model_name": model_name, "device": device}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return HFModel(**kwargs)  # type: ignore[operator]


def parse_slot_overrides(pairs: Iterable[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Slot override must be KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    # Always run from a single case file
    meta: Dict[str, Any] = {}
    scenario_id = "case_file"
    slot_overrides: Dict[str, str] = parse_slot_overrides(args.slot or [])
    if not getattr(args, "case_file", None):
        raise ValueError("--case-file is required")
    scenario, meta = build_case_from_file(Path(args.case_file), slot_overrides, case_id=args.case_id)

    if args.intent_addition:
        scenario.intent_additions = args.intent_addition

    if args.max_spans is not None:
        scenario.spans = scenario.spans[: args.max_spans]

    model = load_model(args.model, device=args.device, torch_dtype=args.torch_dtype)
    # Resolve template pools (scenario defaults + CLI overrides)
    scoring_cfg = scenario.scoring_templates or {}
    pass_templates_default = resolve_template_list(
        scoring_cfg.get("pass")
        or scoring_cfg.get("pass_templates")
        or scoring_cfg.get("pass_template"),
        DEFAULT_PASS_TEMPLATES,
    )
    fail_templates_default = resolve_template_list(
        scoring_cfg.get("fail")
        or scoring_cfg.get("fail_templates")
        or scoring_cfg.get("fail_template"),
        DEFAULT_FAIL_TEMPLATES,
    )

    if args.pass_template:
        pass_templates_used = [t.strip() for t in args.pass_template if t and t.strip()]
    else:
        pass_templates_used = pass_templates_default

    if args.fail_template:
        fail_templates_used = [t.strip() for t in args.fail_template if t and t.strip()]
    else:
        fail_templates_used = fail_templates_default

    scorer = LogProbNormScorer(model, LogProbNormConfig(
        pass_templates=pass_templates_used,
        fail_templates=fail_templates_used,
        tau=args.tau,
    ))
    # Choose scalar per score space (attribution convention: higher is safer)
    scalar_fn = make_scalar(scorer, args.score_space)

    # BASE summaries in probability space for readability
    base = compute_base_scores(scorer, scenario.minimal_context, scenario.spans, scenario.question)
    C_ctx = assemble_context(scenario.minimal_context, scenario.spans, scenario.question)
    fC_scalar = scalar_fn(C_ctx)

    f_without_scalar, f_without_prob, loo_vals = compute_loo(
        scorer,
        scenario.minimal_context,
        scenario.spans,
        scenario.question,
        fC_scalar,
        scalar_fn,
    )
    ranked = sorted(range(len(scenario.spans)), key=lambda i: abs(loo_vals[i]), reverse=True)
    top_k = ranked[: args.top_k if args.top_k is not None else len(ranked)]

    aoi = compute_aoi_baselines(
        scorer,
        scenario.minimal_context,
        scenario.spans,
        scenario.question,
        top_indices=top_k,
        include_empty=args.include_empty_baseline,
        include_minimal=args.include_minimal_baseline,
        scalar_fn=scalar_fn,
    )

    delta = {}

    intent_aoi = compute_intent_aoi(
        scorer,
        scenario.minimal_context,
        scenario.spans,
        scenario.question,
        scenario.intent_additions,
        scalar_fn,
    ) if scenario.intent_additions else {}

    generation_outputs: Dict[str, Any] = {}
    preview_prompts = {
        "Minimal (M)": assemble_context(scenario.minimal_context, [], scenario.question),
        "Full (C)": assemble_context(scenario.minimal_context, scenario.spans, scenario.question),
    }
    for label, prompt in preview_prompts.items():
        gen = generate_outputs(
            model,
            prompt,
            max_new_tokens=args.gen_max_new_tokens,
            num_stochastic=args.num_stochastic_samples,
        )
        if gen is not None:
            generation_outputs[label] = gen

    run_parameters = dict(vars(args))
    run_parameters["slot_overrides"] = slot_overrides

    report: Dict[str, Any] = {
        "meta": {
            "case_id": scenario.case_id,
            "case_title": scenario.title,
            "model_name": getattr(model, "model_name", "unknown"),
            "slots": scenario.slot_values,
            "case_file": str(Path(args.case_file).resolve())
        },
        "base_scores": base,
        "loo": {
            "f_without": f_without_scalar,
            "f_without_prob": f_without_prob,
            "loo": loo_vals,
            "ranked_indices": ranked,
        },
        "aoi": aoi,
        "interaction_gain": delta,
        "intent_aoi": intent_aoi,
        "spans": scenario.spans,
        "question": scenario.question,
        "minimal_context": scenario.minimal_context,
        "scoring_templates": {
            "pass": pass_templates_used,
            "fail": fail_templates_used,
        },
        "intent_additions": list(scenario.intent_additions),
        "model_generations": generation_outputs,
        "run_parameters": run_parameters,
    }
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafetyMargin attribution pipeline.")
    parser.add_argument("--case-file", type=str, required=True, help="Path to a single-case JSON (e.g., dataset/med1.json).")
    parser.add_argument("--case-id", type=str, help="Case identifier to run (defaults to first case).")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name to load.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for HuggingFace model.")
    parser.add_argument("--torch-dtype", type=str, default=None, help="Optional torch dtype (float16, bfloat16, etc).")
    parser.add_argument(
        "--pass-template",
        action="append",
        default=None,
        help="Override refusal templates (repeatable).",
    )
    parser.add_argument(
        "--fail-template",
        action="append",
        default=None,
        help="Override compliance templates (repeatable).",
    )
    parser.add_argument("--tau", type=float, default=1.0, help="Softmax temperature for norm scorer.")
    parser.add_argument(
        "--score-space",
        type=str,
        choices=["prob", "logit"],
        default="logit",
        help="Attribution score space: 'prob' (probability of refusal) or 'logit' (margin/τ). Thresholds interpreted in this space.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="How many spans to display (defaults to all).")
    parser.add_argument("--max-spans", type=int, default=None, help="Maximum number of spans to materialise.")
    parser.add_argument("--slot", action="append", default=[], metavar="KEY=VALUE", help="Override template slot.")
    parser.add_argument("--intent-addition", action="append", default=[], help="Add intent hypothesis span.")
    parser.add_argument("--include-empty-baseline", action="store_true", default=True, help="Include empty baseline AOI.")
    parser.add_argument("--no-empty-baseline", action="store_false", dest="include_empty_baseline", help="Disable empty baseline.")
    parser.add_argument("--include-minimal-baseline", action="store_true", default=True, help="Include minimal baseline AOI.")
    parser.add_argument("--no-minimal-baseline", action="store_false", dest="include_minimal_baseline", help="Disable minimal baseline.")
    parser.add_argument("--gen-max-new-tokens", type=int, default=200, help="Maximum tokens for preview generations (default: 200).")
    parser.add_argument("--num-stochastic-samples", type=int, default=3, help="Number of stochastic samples per context (default: 3).")
    parser.add_argument("--json-output", type=str, help="Optional path to write JSON report.")
    parser.add_argument("--precision", type=int, default=6, help="Decimal places for printed floats (default: 6).")
    # Decision thresholds (interpreted in the chosen score space)
    parser.add_argument("--tau-N", dest="tau_N", type=float, default=0.20, help="Necessity threshold for |LOO| (interpreted in the chosen score space).")
    parser.add_argument("--tau-S", dest="tau_S", type=float, default=0.20, help="Sufficiency threshold for |AOI| (interpreted in the chosen score space).")

    parser.add_argument("--tau-I", dest="tau_I", type=float, default=0.15, help="Intent AOI threshold (interpreted in the chosen score space).")
    parser.add_argument("--tau-zero", dest="tau_zero", type=float, default=0.05, help="Near-zero threshold for ≈0 tests (interpreted in the chosen score space).")
    parser.add_argument("--quiet", action="store_true", help="Suppress textual report, only emit JSON if requested.")
    parser.add_argument("--version", action="version", version="SafetyMargin pipeline 1.0")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    report = run_pipeline(args)
    safe_report = json_safe(report)
    if not args.quiet:
        print_report(report, args)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(safe_report, f, indent=2)
        print(f"\n[pipeline] Wrote JSON report to {output_path.resolve()}")

    meta = report.get("meta", {})
    # Prefer the case-file name (stem) for result tagging so auto-saved
    # reports are clearly tied to the input JSON (e.g. `evidence_fixed_run...`).
    # Fall back to the reported case_id or a generic "scenario" when unavailable.
    try:
        case_file = getattr(args, "case_file", None)
        if case_file:
            scenario_tag = Path(case_file).stem
        else:
            scenario_tag = meta.get("case_id") or "scenario"
    except Exception:
        scenario_tag = meta.get("case_id") or "scenario"
    # sanitize whitespace → underscores and ensure non-empty
    scenario_tag = str(scenario_tag).strip().replace(" ", "_") or "scenario"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_dir = Path("results") / f"{args.model.replace('/', '_')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    auto_path = results_dir / f"{scenario_tag}_run{timestamp}.json"
    with auto_path.open("w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=2)
    if not args.quiet:
        print(f"[pipeline] Auto-saved report to {auto_path.resolve()}")


if __name__ == "__main__":
    main()
