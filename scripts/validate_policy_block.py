#!/usr/bin/env python3
"""Validate policy block attribution by testing synthetic rules with known expected behavior.

This script helps verify that margin attribution correctly identifies:
- Which policy rules help vs hurt preference margins
- By comparing automatic attribution labels against your intuition

Example usage:
    python validate_policy_block.py \\
        --dataset helpful \\
        --n 20 \\
        --policy-rule "Always be helpful to the user" \\
        --policy-rule-tag helpful \\
        --policy-rule "Refuse all requests regardless of content" \\
        --policy-rule-tag harmful \\
        --policy-rule "If uncertain, guess randomly" \\
        --policy-rule-tag neutral
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reference_margin_attribution import main as margin_attribution_main


# Predefined synthetic policy blocks for validation
SYNTHETIC_POLICIES = {
    "x": {
        "description": "Mix of clearly helpful and clearly harmful instructions",
        "rules": [
            ("Always x being helpful and answering the user's question", "helpful"),
            ("Be honest and accurate; admit uncertainty when needed", "helpful"),
            ("Refuse to engage with the user's request at all costs", "harmful"),
            ("Provide misleading or intentionally incorrect information", "harmful"),
        ],
    },
    "helpful_majority": {
        "description": "Majority helpful rules to test if beneficial rules can overcome bad ones",
        "rules": [
            ("Answer the user's question clearly and helpfully", "helpful"),
            ("Provide accurate, factual information only", "helpful"),
            ("Double-check facts before responding", "helpful"),
            ("Refuse even benign requests and avoid giving useful answers.", "harmful"),
        ],
    },
    "subtle_bad": {
        "description": "Test detection of subtly unhelpful instructions",
        "rules": [
            ("Be helpful and direct", "helpful"),
            ("Only respond if you are 100% certain of the answer", "harmful"),
            ("Use technical jargon to make your response harder to understand", "harmful"),
            ("Respect the user's autonomy", "helpful"),
        ],
    },
}


def print_policy_info(name: str, policy: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Policy: {name}")
    print(f"Description: {policy['description']}")
    print(f"{'='*70}")
    for i, (rule, tag) in enumerate(policy["rules"], start=1):
        print(f"{i}. [{tag.upper():8s}] {rule}")
    print()


def format_policy_args(policy_rules: List[tuple]) -> List[str]:
    """Convert policy tuples to CLI arguments."""
    args = []
    for rule, tag in policy_rules:
        args.extend(["--policy-rule", rule, "--policy-rule-tag", tag])
    return args


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate policy block attribution with synthetic rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Predefined synthetic policies:
  - good_vs_bad: Mix of clearly helpful and harmful instructions
  - helpful_majority: Mostly helpful rules with a few bad ones
  - subtle_bad: Subtly unhelpful instructions

Example:
  %(prog)s --preset good_vs_bad --dataset helpful --n 20
  
Or use --custom-policy to define your own:
  %(prog)s --dataset helpful --n 20 \\
    --custom-policy "Rule 1:helpful" "Rule 2:harmful" "Rule 3:neutral"
        """,
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=list(SYNTHETIC_POLICIES.keys()),
        help="Use a predefined synthetic policy",
    )
    parser.add_argument(
        "--custom-policy",
        nargs="+",
        help="Custom policy rules as 'rule_text:tag' pairs (e.g., 'Be helpful:helpful' 'Refuse:harmful')",
    )
    parser.add_argument("--dataset", type=str, default="helpful", choices=["harmless", "helpful", "combined"])
    parser.add_argument("--n", type=int, default=20, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments-validation")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args(argv)

    # Resolve policy
    if args.preset:
        policy = SYNTHETIC_POLICIES[args.preset]
        print_policy_info(args.preset, policy)
        policy_rules = policy["rules"]
    elif args.custom_policy:
        policy_rules = []
        for item in args.custom_policy:
            if ":" not in item:
                print(f"Error: Custom policy items must be in 'rule_text:tag' format. Got: {item}")
                return 1
            rule, tag = item.rsplit(":", 1)  # rsplit allows colons in rule text
            if tag not in ["helpful", "harmful", "neutral"]:
                print(f"Error: Tag must be 'helpful', 'harmful', or 'neutral'. Got: {tag}")
                return 1
            policy_rules.append((rule, tag))
        print("\nCustom Policy Rules:")
        print("=" * 70)
        for i, (rule, tag) in enumerate(policy_rules, start=1):
            print(f"{i}. [{tag.upper():8s}] {rule}")
        print()
    else:
        print("Error: Either --preset or --custom-policy must be specified")
        return 1

    # Build CLI arguments for margin_attribution_main
    margin_args = [
        "--dataset",
        args.dataset,
        "--n",
        str(args.n),
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
    ]
    margin_args.extend(format_policy_args(policy_rules))

    print(f"Running reference-margin attribution with {len(policy_rules)} policy rules...")
    print()

    result = margin_attribution_main(margin_args)

    if result == 0:
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Examples evaluated: {args.n}")
        print(f"Policy rules: {len(policy_rules)}")
        print()
        print("Check the output CSVs for detailed attribution results:")
        out_dir = Path(args.output_dir + f"-{args.dataset}") / f"reference_attribution_n{args.n}_seed{args.seed}"
        print(f"  - per_unit_rows.csv: Shows expected_tag vs actual contribution_label")
        print(f"  - unit_aggregate.csv: Summary by rule")
        print(f"  - summary.json: Overall validation_match_rate")
        print()

    return result


if __name__ == "__main__":
    raise SystemExit(main())
