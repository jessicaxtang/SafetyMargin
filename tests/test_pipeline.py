#!/usr/bin/env python3
"""
Integration test for SafetyMargin pipeline.
Runs a single test case and validates the output structure.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_pipeline(case_file: str, model: str = "meta-llama/Llama-3.2-1B-Instruct", device: str = "cpu") -> dict:
    """
    Run the pipeline on a single case and return the results.
    
    Args:
        case_file: Path to case JSON file
        model: Model name to use
        device: Device (cpu/cuda)
        
    Returns:
        Dictionary with test results
    """
    from scripts.pipeline_nov3 import parse_args, run_pipeline
    
    # Build command-line arguments
    argv = [
        "--case-file", case_file,
        "--model", model,
        "--device", device,
        "--all-pairs",
        "--score-space", "logit",
        "--tau-N", "0.2",
        "--tau-S", "0.2",
        "--tau-D", "0.1",
        "--tau-I", "0.15",
        "--tau-zero", "0.05",
        "--quiet",  # Suppress output for testing
    ]
    
    args = parse_args(argv)
    report = run_pipeline(args)
    
    # Validate structure
    assert "meta" in report, "Missing 'meta' section"
    assert "base_scores" in report, "Missing 'base_scores' section"
    assert "loo" in report, "Missing 'loo' section"
    assert "aoi" in report, "Missing 'aoi' section"
    assert "interaction_gain" in report, "Missing 'interaction_gain' section"
    assert "intent_aoi" in report, "Missing 'intent_aoi' section"
    
    # Validate LOO structure
    loo = report["loo"]
    assert "f_without" in loo, "Missing LOO f_without"
    assert "loo" in loo, "Missing LOO values"
    assert "ranked_indices" in loo, "Missing LOO ranked_indices"
    
    # Validate base scores
    base = report["base_scores"]
    assert "f(M)" in base, "Missing f(M)"
    assert "f(C)" in base, "Missing f(C)"
    assert "m(M)" in base, "Missing m(M)"
    assert "m(C)" in base, "Missing m(C)"
    
    print(f"✓ Test passed for {case_file}")
    print(f"  - Spans: {len(report['spans'])}")
    print(f"  - LOO values computed: {len(loo['loo'])}")
    print(f"  - AOI baselines: {len(report['aoi'])}")
    print(f"  - Interaction pairs: {len(report['interaction_gain'])}")
    print(f"  - Intent additions: {len(report['intent_aoi'])}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Test SafetyMargin pipeline")
    parser.add_argument(
        "--case-file",
        type=str,
        default="dataset/privacy.json",
        help="Case file to test",
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
        "--all",
        action="store_true",
        help="Test all case files",
    )
    
    args = parser.parse_args()
    
    if args.all:
        cases = [
            "dataset/privacy.json",
            "dataset/injection.json",
            "dataset/evidence.json",
            "dataset/hazardous_safety.json",
            "dataset/role_confusion.json",
        ]
        print("Testing all case files...")
        for case in cases:
            try:
                test_pipeline(case, args.model, args.device)
            except Exception as e:
                print(f"✗ Test failed for {case}: {e}")
                return 1
        print("\n✓ All tests passed!")
    else:
        try:
            test_pipeline(args.case_file, args.model, args.device)
            print("\n✓ Test passed!")
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
