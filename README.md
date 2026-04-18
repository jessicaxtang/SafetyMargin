# Optimizing a Margin of Safety via Prompt Repair in Large Language Models

This repository implements an attribution-driven prompt repair workflow that optimizes for a margin of safety in large language models.

## Directory

- `safetymargin/` – Core Python package
  - `models/` – HuggingFace and vLLM model wrappers with teacher-forcing scoring utilities
  - `preference_transfer.py` – Preference margin transfer experiments (core module)
  - `utils.py` – Shared utilities: context assembly, log-prob scoring, generation helpers
  - `sguard_content_filter.py` – Safety content filter wrapper
- `scripts/` – Experiment and analysis scripts
  - `reference_margin_attribution.py` – Main LOO attribution analysis on local datasets
  - `aggregate_reference_attribution.py` – Aggregate results across runs into CSV summaries
  - `generate_variants_and_measure.py` – Sample output variants and measure unsafe rate on local safety datasets
  - `generate_variants_and_measure_hh.py` – Sample output variants on HH-RLHF dataset
  - `analyze_generation_variants.py` – Analyze results from generation variant runs
  - `prompt_edit_success_rate.py` – Measure success rate of prompt edits on toy datasets
  - `validate_policy_block.py` – Validate policy-rule attribution with synthetic rules
  - `plot_all_results.py` – Plot repair effectiveness figures
- `visualization/` – Figure generation scripts
  - `attribution_evaluation.py` – Attribution heatmap and precision/recall evaluation
  - `plot_span_colormap.py` – Color-coded per-unit span attribution view (tornado chart)
  - `plot_revision_margins.py` – Plot margin shifts before/after prompt revisions
  - `figure_common.py` – Shared visualization utilities
- `dataset/` – Datasets in JSON format
  - `toy_data/` – Toy sample dataset for quick testing
  - `handcrafted_v1/`, `handcrafted_v2/` – Custom safety prompt cases with ground-truth labels
- `tests/` – Unit tests

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set your HuggingFace token:
```bash
huggingface-cli login
```

## Reference Margin Attribution

Run LOO attribution analysis on a local dataset:

```bash
python scripts/reference_margin_attribution.py \
  --local-dataset dataset/toy_data/data.json \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --intervention-mode prompt_units \
  --prompt-unit-splitter nltk_sentence \
  --output-dir experiments-toy-data

python scripts/reference_margin_attribution.py \
  --local-dataset dataset/handcrafted_v2/handcrafted_dataset2.json \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --intervention-mode prompt_units \
  --prompt-unit-splitter nltk_sentence \
  --output-dir experiments-handcrafted
```

Aggregate results and generate figures:

```bash
python scripts/aggregate_reference_attribution.py \
  --roots experiments-toy-data experiments-handcrafted

python visualization/attribution_evaluation.py --model-name llama-3.2-1b
python visualization/plot_span_colormap.py \
  --per-unit-path experiments-handcrafted/reference_attribution_n100_seed42_llama-3.2-3b/per_unit_rows.csv \
  --example-index 0
```

## Preference Transfer Experiment

Run preference-margin transfer on HH-RLHF:

```bash
python run_preference_transfer.py \
  --dataset harmless \
  --n 500 \
  --k 5 \
  --temperature 0.7 \
  --base-model meta-llama/Llama-3.2-1B-Instruct

# Faster sampling with vLLM
python run_preference_transfer.py \
  --dataset harmless \
  --n 500 \
  --k 5 \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --sampling-backend vllm

# Pin reference-margin scoring to a second GPU
python run_preference_transfer.py \
  --dataset harmless \
  --n 500 \
  --k 5 \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --sampling-backend vllm \
  --reference-device cuda:1
```

Policy-rule leave-one-out (remove one behavioral rule at a time):

```bash
python run_preference_transfer.py \
  --dataset harmless \
  --n 500 \
  --k 5 \
  --intervention-mode policy_rules \
  --policy-placement prepend \
  --base-model meta-llama/Llama-3.2-1B-Instruct
```

Outputs are written to `experiments-<dataset>/preference_transfer_n<N>_k<K>_seed<S>/`:
- `results.csv` – per-example summary metrics
- `results_detailed.json` – full results with per-unit LOO rows
- `config.json` – run configuration
- `scatter_*.png` – transfer correlation scatter plots
