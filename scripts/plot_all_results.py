#!/usr/bin/env python3
"""
Create comprehensive figure showing repair effectiveness across all 5 attack scenarios.

Generates publication-ready plots for AAAI short paper demonstrating:
1. Before/after safety margins for each scenario
2. Margin improvements across scenarios
3. Generalization to test variations
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load repair transfer results for all 5 scenarios."""
    scenarios = {
        'Privacy': 'privacy_method_transfer.json',
        'Injection': 'injection_method_transfer.json',
        'Evidence': 'evidence_method_transfer.json',
        'Role Confusion': 'role_confusion_method_transfer.json',
        'Hazardous Safety': 'hazardous_safety_method_transfer.json',
    }
    
    results = {}
    for name, filename in scenarios.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: {filepath} not found, skipping {name}")
    
    return results


def plot_repair_effectiveness(results: Dict[str, Dict], output_path: Path):
    """
    Main figure: Before/after margins for all scenarios showing repair effectiveness.
    Clean, publication-ready design with original on left, repaired on right.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = list(results.keys())
    n_scenarios = len(scenarios)
    x_positions = np.arange(n_scenarios)
    
    # Colorblind-safe palette (Wong 2011)
    color_before = '#ffcbcb' #'#D55E00'  # Vermillion (unsafe)
    color_after = '#aadfaa' #'#009E73'   # Bluish green (safe)
    
    # Collect data
    original_margins = []
    test_repaired_means = []
    test_std_errors = []
    minimal_baselines = []
    
    for scenario, data in results.items():
        train = data['train']
        test = data['test']
        original_margins.append(train['original_margin'])
        
        # Calculate mean of test repaired margins
        test_results = test['results']
        test_repaired = [r['repaired_margin'] for r in test_results]
        test_repaired_means.append(np.mean(test_repaired))
        test_std_errors.append(np.std(test_repaired))
        
        minimal_baselines.append(train.get('original_report', {}).get('m(M)', 0))
    
    # Plot original and repaired side-by-side with offset
    width = 0.3
    offset = 0.2
    
    for i, (orig, repair, std_err, minimal) in enumerate(zip(original_margins, test_repaired_means, test_std_errors, minimal_baselines)):
        x_orig = i - offset
        x_repair = i + offset
        
        # Arrow showing improvement direction
        ax.annotate('', xy=(x_repair, repair), xytext=(x_orig, orig),
                   arrowprops=dict(arrowstyle='->', lw=3, color="#202020", alpha=0.7))
        
        # Markers for before/after
        ax.scatter(x_orig, orig, s=100, color=color_before, edgecolor='#D55E00', 
                  linewidth=1.5, zorder=4, marker='o', label='Original prompt $x$' if i == 0 else '')
        ax.scatter(x_repair, repair, s=100, color=color_after, edgecolor='#009E73', 
                  linewidth=1.5, zorder=4, marker='o', label='Repaired prompt $x\'$' if i == 0 else '')
        
        # Error bar for test variation (if available)
        if std_err > 0:
            ax.errorbar(x_repair, repair, yerr=std_err, fmt='none', 
                       ecolor="#5B5B5B", elinewidth=2, capsize=4, 
                       capthick=2, alpha=0.7, zorder=3)
        
        # Minimal baseline indicator (subtle horizontal line)
        ax.plot([x_orig - width, x_repair + width], [minimal, minimal], 
                color='#5B5B5B', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Improvement annotation (between the two points)
        improvement = repair - orig
        if std_err > 0:
            label_text = f'+{improvement:.2f} ± {std_err:.2f}'
        else:
            label_text = f'+{improvement:.2f}'

        label_text = '' # don't plot labels to reduce clutter

        mid_x = (x_orig + x_repair) / 2
        mid_y = (orig + repair) / 2
        ax.annotate(label_text, 
                   xy=(mid_x, mid_y), xytext=(0, 10),
                   textcoords='offset points', ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.8))
    
    # Zero reference line
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=1.2, alpha=0.4, zorder=1)
    
    # Styling
    # ax.set_xlabel('Attack Scenario', fontsize=13)
    ax.set_ylabel('Directional Safety Margin $m(C)$', fontsize=15, fontfamily='Times New Roman')
    # ax.set_title('Repair Effectiveness on Test Cases', fontsize=14, pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios, fontsize=15, fontfamily='Times New Roman')
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to start from min baseline with padding
    y_min = min(min(original_margins), min(minimal_baselines)) * 1.1
    y_max = max([r + e for r, e in zip(test_repaired_means, test_std_errors)]) * 1.15
    ax.set_ylim(y_min, y_max)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    legend_elements = handles + [
        Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5,
               label='Minimal-context baseline $m(M)$')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=False, framealpha=0.95, edgecolor='gray', prop={'family': 'Times New Roman', 'size': 13})
    
    plt.tight_layout()
    
    # Save PNG only
    plt.savefig(output_path / 'repair_effectiveness.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: repair_effectiveness.png")
    plt.close()

def print_summary_statistics(results: Dict[str, Dict]):
    """Print comprehensive statistics for paper."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    all_train_improvements = []
    all_test_improvements = []
    all_success_rates = []
    
    for scenario, data in results.items():
        print(f"\n{scenario}:")
        print(f"  Train: m(C) {data['train']['original_margin']:.2f} → "
              f"{data['train']['repaired_margin']:.2f} (Δm = {data['train']['improvement']:+.2f})")
        print(f"  Test:  {data['test']['n_cases']} cases, "
              f"mean Δm = {data['test']['mean_improvement']:+.2f} ± {data['test']['std_improvement']:.2f}")
        print(f"  Success rate: {data['test']['success_rate']:.0f}% "
              f"({data['test']['success_count']}/{data['test']['n_repaired']} improved)")
        print(f"  Generalization: {data['generalization_quality']}")
        
        all_train_improvements.append(data['train']['improvement'])
        all_test_improvements.append(data['test']['mean_improvement'])
        all_success_rates.append(data['test']['success_rate'])
    
    print("\n" + "-"*80)
    print("AGGREGATE STATISTICS:")
    print(f"  Mean train improvement: {np.mean(all_train_improvements):.3f} ± {np.std(all_train_improvements):.3f}")
    print(f"  Mean test improvement:  {np.mean(all_test_improvements):.3f} ± {np.std(all_test_improvements):.3f}")
    print(f"  Mean success rate: {np.mean(all_success_rates):.1f}% ± {np.std(all_success_rates):.1f}%")
    print(f"  Scenarios with >80% success: {sum(1 for r in all_success_rates if r > 80)}/{len(all_success_rates)}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive figures for all repair transfer results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing *_method_transfer.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save plots",
    )
    
    args = parser.parse_args()
    
    # Load all results
    print("Loading results from all scenarios...")
    results = load_all_results(args.results_dir)
    
    if not results:
        print("No results found! Make sure *_method_transfer.json files exist in results/")
        return
    
    print(f"Found results for {len(results)} scenarios: {', '.join(results.keys())}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    plot_repair_effectiveness(results, args.output_dir)
    # plot_repair_effectiveness_slope(results, args.output_dir)
    # plot_generalization_summary(results, args.output_dir)
    # plot_span_consistency(results, args.output_dir)
    # generate_summary_table(results, args.output_dir)
    
    # Print summary
    print_summary_statistics(results)
    
    print(f"\n✅ All figures saved to: {args.output_dir}")
    print("\nFigures for paper:")
    print(f"  - Main figure: repair_effectiveness_main.png")
    # print(f"  - Alternative (slope): repair_effectiveness_slope.png")
    print(f"  - Generalization: generalization_summary.png")
    print(f"  - Consistency: span_consistency.png")
    print(f"  - Table: results_table.tex")


if __name__ == "__main__":
    main()
