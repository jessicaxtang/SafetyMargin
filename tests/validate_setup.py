#!/usr/bin/env python3
"""
Quick validation script to ensure the SafetyMargin codebase is properly set up.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")
    
    try:
        from safetymargin.datasets.scenarios import Scenario, build_case_from_file, normalise_user_question
        print("✓ conflection.datasets.scenarios")
    except Exception as e:
        print(f"✗ conflection.datasets.scenarios: {e}")
        return False
    
    try:
        import importlib.util
        spec = importlib.util.find_spec("scripts.pipeline")
        if spec is not None:
            print("✓ scripts.pipeline (module found)")
        else:
            print("✗ scripts.pipeline (module not found)")
            return False
    except Exception as e:
        print(f"⚠ scripts.pipeline: {e}")
    
    return True


def check_data_files():
    """Check that all scenario files exist."""
    print("\nChecking data files...")
    
    required_files = [
        "dataset/privacy.json",
        "dataset/injection.json",
        "dataset/evidence.json",
        "dataset/hazardous_safety.json",
        "dataset/role_confusion.json",
        "dataset/references.json",
        "dataset/intents.json",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = ROOT / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def check_scenario_loading():
    """Test loading a scenario file."""
    print("\nTesting scenario loading...")
    
    try:
        from safetymargin.datasets.scenarios import build_case_from_file
        
        case_path = ROOT / "dataset" / "privacy.json"
        scenario, meta = build_case_from_file(case_path)
        
        print(f"✓ Loaded scenario: {scenario.case_id}")
        print(f"  - Spans: {len(scenario.spans)}")
        print(f"  - Minimal context: {len(scenario.minimal_context)}")
        print(f"  - Intent additions: {len(scenario.intent_additions)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load scenario: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("SafetyMargin Codebase Validation")
    print("=" * 60)
    
    results = []
    
    # Run checks
    results.append(("Imports", check_imports()))
    results.append(("Data Files", check_data_files()))
    results.append(("Scenario Loading", check_scenario_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! The codebase is ready to use.")
        print("\nNext steps:")
        print("  1. Set your HuggingFace token: export HUGGINGFACE_TOKEN='your_token'")
        print("  2. Run a single test: python3 scripts/test_pipeline.py --device cpu")
        print("  3. Run all experiments: ./scripts/run_all_experiments.sh")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
