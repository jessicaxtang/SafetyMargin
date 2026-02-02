"""Report generation and formatting for SafetyMargin pipeline."""

from statistics import median
from typing import Any, Dict, List, Optional

import argparse


def format_section(title: str) -> str:
    """Format a section header for console output."""
    return f"\n=== {title.upper()} ==="


def print_report(report: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print a comprehensive attribution report to console.
    
    Args:
        report: Dictionary containing attribution results, metadata, and scores
        args: Command-line arguments containing thresholds and display options
    """
    meta = report["meta"]
    spans = report["spans"]
    base = report["base_scores"]
    loo = report["loo"]
    aoi = report["aoi"]
    intent_aoi = report["intent_aoi"]
    scoring_templates = report.get("scoring_templates") or {}
    intent_additions = report.get("intent_additions") or list(intent_aoi.keys())
    
    # Create label map handling both string and dict formats
    intent_label_map = {}
    for idx, addition in enumerate(intent_additions):
        if isinstance(addition, dict):
            key = addition.get("text", str(addition))
        else:
            key = addition
        intent_label_map[key] = f"a_{idx}"

    # Header section
    print(format_section(f"SCENARIO {meta['case_id']}"))
    print(f"Model: {meta['model_name']}")
    print("Question:", report["question"])
    print(f"Attribution space: {args.score_space}")
    print(
        f"Thresholds (space={args.score_space}): τ_N={args.tau_N}, τ_S={args.tau_S}, "
        f"τ_I={args.tau_I}, τ_zero={args.tau_zero}"
    )

    # Teacher-forced targets
    print(format_section("TEACHER-FORCED TARGETS"))
    pass_templates = scoring_templates.get("pass") or []
    fail_templates = scoring_templates.get("fail") or []
    print("Refusal (safe) templates:")
    if pass_templates:
        for tpl in pass_templates:
            print(f"  - {tpl}")
    else:
        print("  (none)")
    print("Compliance (unsafe) templates:")
    if fail_templates:
        for tpl in fail_templates:
            print(f"  - {tpl}")
    else:
        print("  (none)")

    # Model generations
    model_generations = report.get("model_generations") or {}
    if model_generations:
        print(format_section("MODEL GENERATIONS"))
        for label, bundle in model_generations.items():
            greedy = bundle.get("greedy") or "(no output)"
            print(f"{label} – Greedy:\n{greedy}")
            for j, sample in enumerate(bundle.get("stochastic") or [], start=1):
                text = sample or "(no output)"
                print(f"{label} – Sample {j}:\n{text}")

    # Context components
    print("\nM: Minimal context:")
    for line in report["minimal_context"]:
        print("  -", line)
    print()
    print("S: Specifications:")
    for idx, span in enumerate(spans):
        print(f"  s_{idx}: {span}")
    print()
    if intent_additions:
        print("A: Intent Additions:")
        for idx, addition in enumerate(intent_additions):
            # Handle both string and dict formats
            if isinstance(addition, dict):
                addition_text = addition.get("text", str(addition))
            else:
                addition_text = addition
            print(f"  a_{idx}: {addition_text}")
        print()

    # Format helper
    prec = max(0, int(getattr(args, "precision", 6)))
    fmt = lambda x: f"{x:+0.{prec}f}"
    
    # Base scores
    print(format_section("BASE SCORES"))
    print(f"f(M) = {base['f(M)']:.{prec}f} | m(M) = {base['m(M)']:.{prec}f}")
    print(f"f(C) = {base['f(C)']:.{prec}f} | m(C) = {base['m(C)']:.{prec}f}")
    print(f"Attribution space: {args.score_space} (margin/τ when logit)")

    # LOO necessity
    print(format_section("LOO NECESSITY"))
    loo_vals = loo["loo"]
    tau_N = float(getattr(args, "tau_N", 1.0))
    sorted_loo_indices = sorted(range(len(spans)), key=lambda i: abs(loo_vals[i]), reverse=True)
    for idx in sorted_loo_indices:
        value = loo_vals[idx]
        without_prob = loo.get("f_without_prob", loo.get("f_without", []))[idx]
        direction = "↑" if value > 0 else ("↓" if value < 0 else "·")
        if value > +tau_N:
            verdict = "keep"
        elif value < -tau_N:
            verdict = "remove/rewrite"
        else:
            verdict = "neutral"
        print(f"s_{idx}: LOO={fmt(value)} {direction}  → {verdict} | f(C\\s_{idx})={without_prob:.{prec}f}")

    # AOI sufficiency
    print(format_section("AOI SUFFICIENCY"))
    print(f"Quick Glance (AOI[M], space={args.score_space}):")
    
    def aoi_m_value(i: int) -> float:
        val = aoi.get(i, {}).get("M")
        return float("-inf") if val is None else float(val)

    sorted_aoi_indices = sorted(range(len(spans)), key=aoi_m_value, reverse=True)

    for idx in sorted_aoi_indices:
        aoi_m = aoi.get(idx, {}).get("M")
        if aoi_m is not None:
            print(f"s_{idx}: AOI[M] = {fmt(aoi_m)}")
    print()
    print("Detailed View:")
    print("Note: AOI[C\\{s_i}] equals the LOO in the chosen space.")
    for idx in sorted_aoi_indices:
        print(f"s_{idx}: {spans[idx]}")
        for bname, val in aoi.get(idx, {}).items():
            print(f"  AOI[{bname}] = {fmt(val)}")

    # Intent AOI
    if intent_aoi:
        print(format_section("INTENT AOI ADDITIONS"))
        # Sort by ΔM(C) magnitude (or ΔM(M) if ΔM(C) not available)
        sorted_items = sorted(
            intent_aoi.items(),
            key=lambda kv: abs(kv[1].get("ΔM(C)", kv[1].get("ΔM(M)", 0))),
            reverse=True
        )
        for addition, metrics in sorted_items:
            label = intent_label_map.get(addition, addition)
            print(f"{label}:")
            for metric_name, val in metrics.items():
                print(f"  {metric_name} = {fmt(val)}")

    # Decision labels
    print(format_section("DECISION LABELS"))
    tau_zero = float(getattr(args, "tau_zero", 0.5))
    tau_S = float(getattr(args, "tau_S", 1.0))
    tau_N = float(getattr(args, "tau_N", 1.0))
    tau_D = float(getattr(args, "tau_D", 1.0))
    tau_I = float(getattr(args, "tau_I", 1.0))

    baseline_names = set()
    for i in aoi.keys():
        baseline_names.update(aoi[i].keys())

    # Helper functions
    def get_aoi(i: int, name: str) -> Optional[float]:
        return aoi.get(i, {}).get(name)

    def parse_targeted(name: str) -> Optional[int]:
        """Map targeted baseline keys to j indices: 'M+{s_j}'"""
        if name.startswith("M+{") and name.endswith("}"):
            inner = name[3:-1]
            if inner.startswith("s_"):
                try:
                    return int(inner[2:])
                except Exception:
                    return None
        return None

    # Span-level labels
    for i, span in enumerate(spans):
        labels: List[str] = []
        a_empty = get_aoi(i, "∅")
        a_min = get_aoi(i, "M")
        a_loo = get_aoi(i, "C\\{s_i}")

        # Dependency: α_i(M) ≈ 0 but α_i(M∪{s_j}) ≫ 0
        if a_min is not None and abs(a_min) <= tau_zero:
            for b in aoi.get(i, {}):
                j = parse_targeted(b)
                if j is None:
                    continue
                val = aoi[i][b]
                if abs(val) > tau_S:
                    labels.append(f"Dependency on s_{j}")

        # Graded primary label
        loo_val = loo["loo"][i]
        if (abs(loo_val) < tau_N) and (a_min is not None and abs(a_min) < tau_S):
            labels.append("Neutral")
        else:
            if (loo_val > +tau_N) or (a_min is not None and a_min > +tau_S):
                labels.append("Safety-helping (keep)")
            if (loo_val < -tau_N) or (a_min is not None and a_min < -tau_S):
                labels.append("Risk-increasing (candidate for removal)")

        # Conflict/jailbreak signal
        if a_min is not None and a_min < -tau_S:
            labels.append("Conflict/Jailbreak (risk on M)")

        # Shortcut suspicion
        if a_min is not None and a_min > tau_S:
            for b in aoi.get(i, {}):
                j = parse_targeted(b)
                if j is None:
                    continue
                val = aoi[i][b]
                if abs(val) <= tau_zero and j != i:
                    labels.append(f"Shortcut suspicion (collapses with s_{j})")
                    break

        if labels:
            print(f"s_{i}: {spans[i]}")
            for lab in sorted(set(labels)):
                print(f"  - {lab}")



    # Intent analysis
    if intent_aoi:
        print("\nIntent:")
        intents = list(intent_aoi.items())
        # Sort by ΔM(C) magnitude (or ΔM(M) if ΔM(C) not available)
        intents_sorted = sorted(
            intents,
            key=lambda kv: abs(kv[1].get("ΔM(C)", kv[1].get("ΔM(M)", 0))),
            reverse=True
        )
        top_key, top_metrics = intents_sorted[0]
        top_val = top_metrics.get("ΔM(C)", top_metrics.get("ΔM(M)", 0))
        if abs(top_val) > tau_I:
            dir_lab = "Aligned Intent" if top_val > 0 else "Misaligned Intent"
            print(f"{dir_lab}: {top_key} (|ΔM(C)|={abs(top_val):.{prec}f})")
            print("Note: stability across paraphrases not assessed in this run.")
        else:
            print("Ambiguous Intent: |ΔM(C)| below threshold or margins small.")

    # Repair suggestions
    repair_suggestions: List[str] = []
    review_flags: List[str] = []
    for i, span in enumerate(spans):
        span_reasons: List[str] = []
        loo_val = loo["loo"][i]
        aoi_min = get_aoi(i, "M")
        risk_loo_strong = bool(loo_val < -tau_N)
        risk_loo_soft = bool(loo_val < -tau_N)
        risk_aoi = bool(aoi_min is not None and aoi_min < -tau_S)
        if risk_loo_soft:
            span_reasons.append(f"LOO={fmt(loo_val)}")
        if risk_aoi:
            span_reasons.append(f"AOI(M)={fmt(aoi_min)} < -τ_S")
        if risk_loo_strong and risk_aoi:
            joined = "; ".join(span_reasons)
            repair_suggestions.append(
                f"STRONG: Remove or rewrite s_{i} ({span}) [{joined}] "
                f"and re-run to ensure f(C) increases (safer)."
            )
        elif (risk_loo_soft ^ risk_aoi) or (risk_loo_soft and aoi_min is None):
            joined = "; ".join(span_reasons)
            review_flags.append(
                f"REVIEW: Flag s_{i} ({span}) for human review [{joined}]"
            )

    if intent_aoi:
        intents_sorted = sorted(
            intent_aoi.items(),
            key=lambda kv: abs(kv[1].get("ΔM(C)", kv[1].get("ΔM(M)", 0))),
            reverse=True
        )
        top_key, top_metrics = intents_sorted[0]
        top_val = top_metrics.get("ΔM(C)", top_metrics.get("ΔM(M)", 0))
        if top_val < -tau_I:
            repair_suggestions.append(
                f"Counter or weaken '{top_key}' which appears risk-increasing "
                f"(ΔM(C)={fmt(top_val)}). Re-run to verify improvement."
            )

    print(format_section("REPAIR SUGGESTIONS"))
    if repair_suggestions:
        for suggestion in repair_suggestions:
            print(f"- {suggestion}")
    if review_flags:
        if not repair_suggestions:
            pass  # Header already printed
        for item in review_flags:
            print(f"- {item}")
    if not repair_suggestions and not review_flags:
        print("No high-risk spans or intent flags above thresholds; no repairs suggested.")
