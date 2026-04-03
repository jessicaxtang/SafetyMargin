
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

# Set the font family to 'serif' and specify 'Times New Roman' as the preferred serif font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

rng = np.random.default_rng(0)

# List of model names and corresponding result files
model_files = {
    "llama-3.2-1b": "generation_variants_local2mar30_llama-3.2-1b.json",
    "llama-3.2-3b": "generation_variants_local2mar30_llama-3.2-3b.json",
    "llama-3-8b": "generation_variants_local2mar30_llama-3-8b.json",
    "qwen-2.5-1.5b": "generation_variants_local2mar30_qwen-2.5-1.5b.json",
    "qwen-2.5-3b": "generation_variants_local2mar30_qwen-2.5-3b.json",
    "qwen-2.5-7b": "generation_variants_local2mar30_qwen-2.5-7b.json"
}
model_order = [
    "llama-3.2-1b",
    "llama-3.2-3b",
    "llama-3-8b",
    "qwen-2.5-1.5b",
    "qwen-2.5-3b",
    "qwen-2.5-7b",
]

labels = ["original", "remove_unsafe"]
label_display = {"original": "original", "remove_unsafe": "edited"}
colour_display = {"original": "#F4785C", "remove_unsafe": "#55A868"}
results_dir = "results_generate_var_n=5" #"results_generate_var_n_measure"

all_means = {variant: [] for variant in labels}
all_errors = {variant: [] for variant in labels}
table = []
processed_models = []

def bootstrap_ci(values, n_boot=2000, ci=0.95):
    if len(values) == 0:
        return (0.0, 0.0)
    values = np.asarray(values)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lower = np.percentile(samples, (1 - ci) / 2 * 100)
    upper = np.percentile(samples, (1 + ci) / 2 * 100)
    return lower, upper

for model_name in model_order:
    filename = model_files[model_name]
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue
    processed_models.append(model_name)
    data = json.load(open(path))
    df = pd.DataFrame(data)
    grouped = df.groupby(["variant", "example_index"])
    summary = defaultdict(dict)
    for (variant, ex_idx), group in grouped:
        valid = group["unsafe_any"].dropna()
        unsafe_rate = float(valid.mean()) if not valid.empty else np.nan
        summary[variant][ex_idx] = {"unsafe_rate": unsafe_rate}
    variant_stats = {}
    for v, ex in summary.items():
        rates = [d["unsafe_rate"] for d in ex.values() if not np.isnan(d["unsafe_rate"])]
        if not rates:
            continue
        mean_val = float(np.mean(rates))
        lower, upper = bootstrap_ci(rates)
        variant_stats[v] = {
            "mean": mean_val,
            "err_low": mean_val - lower,
            "err_high": upper - mean_val,
        }

    table.append({"model": model_name, "stats": {l: variant_stats.get(l) for l in labels}})
    for i, l in enumerate(labels):
        stats = variant_stats.get(l)
        if stats:
            all_means[l].append(stats["mean"])
            all_errors[l].append((stats["err_low"], stats["err_high"]))
        else:
            all_means[l].append(0)
            all_errors[l].append((0, 0))

# Plot: Each model as a group, bars for each variant
if not processed_models:
    raise SystemExit("No generation result files were found; aborting plot.")

bar_width = 0.25
index = np.arange(len(processed_models))
group_offset = bar_width * (len(labels) - 1) / 2

fig, ax = plt.subplots(figsize=(8, 5))
bar_containers = []
for i, l in enumerate(labels):
    # yerr = np.array(all_errors[l]).T if all_errors[l] else None
    bars = ax.bar(
        index + i * bar_width,
        all_means[l],
        bar_width,
        label=label_display[l],
        # yerr=yerr,
        capsize=4,
        color=colour_display.get(l),
    )

    bar_containers.append(bars)

for bars in bar_containers:
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

ax.set_xlabel("Model", fontsize=15)
ax.set_ylabel("Mean unsafe rate", fontsize=15)
ax.set_title("Unsafe rate by prompt variant across models", fontsize=15)
ax.set_xticks(index + group_offset, processed_models)
max_val = max((value for values in all_means.values() for value in values), default=0)
y_max = max(0.5, max_val + 0.02)
ax.set_ylim(0, y_max)
ax.legend()
fig.tight_layout()
save_path = os.path.join(results_dir, "unsafe_rate_by_variant_across_models.png")
plt.savefig(save_path, dpi=200)
print(f"plot saved to {save_path}")

# Print summary table
print("\nUnsafe rate by variant across models:")
header = ["Model"] + [label_display[l] for l in labels]
print("\t".join(header))
for row in table:
    values = []
    for l in labels:
        stats = row["stats"].get(l)
        if stats:
            low = stats["mean"] - stats["err_low"]
            high = stats["mean"] + stats["err_high"]
            values.append(f"{stats['mean']:.3f} [{low:.3f}, {high:.3f}]")
        else:
            values.append("n/a")
    print("\t".join([row["model"]] + values))
