#!/usr/bin/env python3
"""
Test repair method generalization: Does the attribution+repair method work consistently across test variations?

Workflow:
1. Run FULL attribution pipeline on training case → identify + remove risk spans → measure improvement
2. Run SAME METHOD on each test variation independently → measure improvements
3. Compare repair effectiveness: training vs test (mean, std, success rate)
4. Shows method generalizes, not just specific span indices
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev, median
from typing import Dict, List, Any, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pipeline import parse_args as pipeline_parse_args, run_pipeline


def identify_risk_spans(report: Dict, tau_N: float = 0.2, tau_S: float = 0.2) -> List[int]:
    """Identify spans flagged for removal based on LOO and AOI.
    
    A span is risky if:
    - LOO < -tau_N (removing it helps), OR
    - AOI(M) < -tau_S (adding it to minimal context hurts)
    """
    loo_vals = report["loo"]["loo"]
    aoi = report["aoi"]
    
    risk_spans = []
    for i in range(len(loo_vals)):
        loo_val = loo_vals[i]
        aoi_M = aoi.get(i, {}).get("M")
        
        # Flag for removal if LOO suggests removal OR AOI suggests it's harmful
        is_loo_risky = loo_val < -tau_N
        is_aoi_risky = aoi_M is not None and aoi_M < -tau_S
        
        if is_loo_risky or is_aoi_risky:
            risk_spans.append(i)
    
    return risk_spans


def create_repaired_case(case: Dict, risk_span_indices: List[int]) -> Dict:
    """Create a repaired version of the case with risk spans removed and protective intents added."""
    repaired = case.copy()
    
    # Remove risk spans
    original_spans = case["spans"]
    repaired["spans"] = [s for i, s in enumerate(original_spans) if i not in risk_span_indices]
    
    # Add protective intent additions (if they exist in the case)
    if "intent_additions" in case and case["intent_additions"]:
        protective_intents = [
            intent for intent in case["intent_additions"]
            if intent.get("expected_outcome") == "safe"
        ]
        if protective_intents:
            # Append protective intents to the spans
            repaired["spans"].extend(protective_intents)
    
    # Update case_id
    repaired["case_id"] = case["case_id"] + "_REPAIRED"
    
    return repaired


def run_repair_on_case(case_file: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run full attribution + repair on a single case."""
    # Step 1: Run attribution
    pipeline_args = pipeline_parse_args([
        "--case-file", str(case_file),
        "--model", args.model,
        "--device", args.device,
        "--score-space", args.score_space,
        "--quiet",
    ])
    original_report = run_pipeline(pipeline_args)
    original_margin = original_report["base_scores"]["m(C)"]
    
    # Step 2: Identify risk spans
    risk_spans = identify_risk_spans(original_report, args.tau_N, args.tau_S)
    
    if not risk_spans:
        return {
            "original_margin": original_margin,
            "repaired_margin": original_margin,
            "improvement": 0.0,
            "risk_spans": [],
            "removed_spans": [],
            "added_intents": [],
            "n_spans_removed": 0,
            "status": "no_repair_needed",
        }
    
    # Step 3: Create and test repaired prompt
    with open(case_file, "r") as f:
        data = json.load(f)
    original_case = data["cases"][0]
    
    # Extract removed span texts
    removed_spans = []
    for idx in risk_spans:
        span = original_case["spans"][idx]
        removed_spans.append({
            "index": idx,
            "id": span.get("id"),
            "label": span.get("label"),
            "text": span.get("text"),
            "tags": span.get("tags", []),
        })
    
    # Extract added intent texts
    added_intents = []
    if "intent_additions" in original_case and original_case["intent_additions"]:
        for intent in original_case["intent_additions"]:
            if intent.get("expected_outcome") == "safe":
                added_intents.append({
                    "id": intent.get("id"),
                    "label": intent.get("label"),
                    "text": intent.get("text"),
                    "expected_outcome": intent.get("expected_outcome"),
                })
    
    repaired_case = create_repaired_case(original_case, risk_spans)
    temp_file = Path(f"/tmp/repaired_{case_file.name}")
    with open(temp_file, "w") as f:
        json.dump({"cases": [repaired_case]}, f)
    
    repaired_args = pipeline_parse_args([
        "--case-file", str(temp_file),
        "--model", args.model,
        "--device", args.device,
        "--score-space", args.score_space,
        "--quiet",
    ])
    repaired_report = run_pipeline(repaired_args)
    repaired_margin = repaired_report["base_scores"]["m(C)"]
    
    improvement = repaired_margin - original_margin
    
    return {
        "original_margin": original_margin,
        "repaired_margin": repaired_margin,
        "improvement": improvement,
        "risk_spans": risk_spans,
        "removed_spans": removed_spans,
        "added_intents": added_intents,
        "n_spans_removed": len(risk_spans),
        "status": "repaired",
        "f_C_original": original_report["base_scores"]["f(C)"],
        "f_C_repaired": repaired_report["base_scores"]["f(C)"],
    }


def test_repair_transfer(train_file: Path, test_file: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Test if repair method generalizes from training to test variations."""
    
    print("\n" + "="*80)
    print("REPAIR METHOD GENERALIZATION TEST")
    print("="*80)
    print("\nApplying SAME METHOD (thresholds, attribution) to train and test independently")
    
    # Step 1: Get training results (from cache or compute)
    if args.train_results and Path(args.train_results).exists():
        print(f"\n[1/3] Loading cached TRAINING results: {args.train_results}")
        with open(args.train_results, "r") as f:
            cached = json.load(f)
            # Extract first result if batch validation was run
            if isinstance(cached, list):
                train_result = cached[0]
            else:
                train_result = cached
        
        # Normalize keys from validate_repairs.py format
        if 'margin_improvement' in train_result:
            train_result['improvement'] = train_result['margin_improvement']
        if 'n_spans_removed' not in train_result and 'risk_spans' in train_result:
            train_result['n_spans_removed'] = len(train_result['risk_spans'])
        
        print(f"  Loaded from cache")
    else:
        print(f"\n[1/3] Testing method on TRAINING: {train_file.name}")
        train_result = run_repair_on_case(train_file, args)
    
    print(f"  Original m(C) = {train_result['original_margin']:.4f}")
    if train_result['status'] == 'repaired':
        print(f"  Identified {len(train_result['risk_spans'])} risk spans: {train_result['risk_spans']}")
        print(f"  Repaired m(C) = {train_result['repaired_margin']:.4f}")
        print(f"  Improvement Δm = {train_result['improvement']:+.4f}")
    else:
        print(f"  No repair needed")
    
    # Step 2: Load TEST variations
    print(f"\n[2/3] Loading TEST variations: {test_file.name}")
    with open(test_file, "r") as f:
        test_data = json.load(f)
    test_cases = test_data["cases"]
    print(f"  Found {len(test_cases)} test variations")
    
    # Step 3: Run repair method on EACH test case independently
    print(f"\n[3/3] Testing method on each TEST variation...")
    test_results = []
    
    temp_dir = Path("/tmp/repair_transfer")
    temp_dir.mkdir(exist_ok=True)
    
    for i, test_case in enumerate(test_cases, 1):
        # Create temp file for this test case
        temp_file = temp_dir / f"test_case_{i}.json"
        with open(temp_file, "w") as f:
            json.dump({"cases": [test_case]}, f)
        
        # Run full attribution + repair
        print(f"  Test {i}/{len(test_cases)}: {test_case.get('case_id', f'case_{i}')}")
        result = run_repair_on_case(temp_file, args)
        result["case_id"] = test_case.get("case_id", f"test_{i}")
        test_results.append(result)
        
        print(f"    Original m(C) = {result['original_margin']:.4f}")
        if result['status'] == 'repaired':
            print(f"    Risk spans: {result['risk_spans']} ({result['n_spans_removed']} removed)")
            print(f"    Repaired m(C) = {result['repaired_margin']:.4f}")
            print(f"    Improvement Δm = {result['improvement']:+.4f}")
        else:
            print(f"    No repair needed")
    
    # Step 4: Compare effectiveness
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS")
    print("="*80)
    
    # Filter repaired cases
    train_repaired = train_result['status'] == 'repaired'
    test_repaired = [r for r in test_results if r['status'] == 'repaired']
    
    print(f"\nCases requiring repair:")
    print(f"  Training: {'Yes' if train_repaired else 'No'}")
    print(f"  Test: {len(test_repaired)}/{len(test_results)}")
    
    if train_repaired and test_repaired:
        # Success rates
        train_success = train_result['improvement'] > 0
        test_improvements = [r['improvement'] for r in test_repaired]
        test_success_count = sum(1 for imp in test_improvements if imp > 0)
        test_success_rate = test_success_count / len(test_repaired) * 100
        
        print(f"\n--- REPAIR SUCCESS RATES ---")
        print(f"Training: {'✅' if train_success else '❌'} ({train_result['improvement']:+.4f})")
        print(f"Test: {test_success_count}/{len(test_repaired)} ({test_success_rate:.0f}%)")
        
        # Improvement statistics
        print(f"\n--- MARGIN IMPROVEMENTS ---")
        print(f"Training Δm: {train_result['improvement']:+.4f}")
        print(f"Test Δm (mean): {mean(test_improvements):+.4f} ± {stdev(test_improvements) if len(test_improvements) > 1 else 0:.4f}")
        print(f"Test Δm (median): {median(test_improvements):+.4f}")
        print(f"Test Δm range: [{min(test_improvements):+.4f}, {max(test_improvements):+.4f}]")
        
        # Spans removed
        test_spans_removed = [r['n_spans_removed'] for r in test_repaired]
        print(f"\n--- SPANS REMOVED ---")
        print(f"Training: {train_result['n_spans_removed']}")
        print(f"Test (mean): {mean(test_spans_removed):.1f} ± {stdev(test_spans_removed) if len(test_spans_removed) > 1 else 0:.1f}")
        print(f"Test (median): {median(test_spans_removed):.0f}")
        
        # Overall assessment
        if test_success_rate == 100:
            quality = "strong"
            print("\n✅ STRONG GENERALIZATION: Method works on 100% of test cases")
        elif test_success_rate >= 80:
            quality = "good"
            print(f"\n✅ GOOD GENERALIZATION: Method works on {test_success_rate:.0f}% of test cases")
        elif test_success_rate >= 50:
            quality = "moderate"
            print(f"\n⚠️  MODERATE GENERALIZATION: Method works on {test_success_rate:.0f}% of test cases")
        else:
            quality = "poor"
            print(f"\n❌ POOR GENERALIZATION: Method only works on {test_success_rate:.0f}% of test cases")
        
        return {
            "status": "completed",
            "train_file": str(train_file),
            "test_file": str(test_file),
            "train": train_result,
            "test": {
                "n_cases": len(test_results),
                "n_repaired": len(test_repaired),
                "results": test_results,
                "success_count": test_success_count,
                "success_rate": test_success_rate,
                "mean_improvement": mean(test_improvements),
                "std_improvement": stdev(test_improvements) if len(test_improvements) > 1 else 0,
                "median_improvement": median(test_improvements),
                "mean_spans_removed": mean(test_spans_removed),
            },
            "generalization_quality": quality,
        }
    else:
        print("\n⚠️  Insufficient repairs to assess generalization")
        return {
            "status": "insufficient_data",
            "train_file": str(train_file),
            "test_file": str(test_file),
            "train": train_result,
            "test": {
                "n_cases": len(test_results),
                "results": test_results,
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test if repairs identified on training data transfer to test variations"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Training scenario file (e.g., dataset/privacy.json)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Test variations file (e.g., dataset/privacy_test.json)",
    )
    parser.add_argument(
        "--train-results",
        type=Path,
        help="Pre-computed training results from validate_repairs.py (avoids re-running)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)",
    )
    parser.add_argument(
        "--score-space",
        type=str,
        choices=["prob", "logit"],
        default="logit",
        help="Attribution space",
    )
    parser.add_argument(
        "--tau-N",
        type=float,
        default=0.2,
        help="LOO necessity threshold",
    )
    parser.add_argument(
        "--tau-S",
        type=float,
        default=0.2,
        help="AOI sufficiency threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output file",
    )
    
    args = parser.parse_args()
    
    # Run transfer test
    results = test_repair_transfer(args.train_file, args.test_file, args)
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
