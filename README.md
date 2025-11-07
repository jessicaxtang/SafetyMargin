# Editing Prompts to Optimize a Margin of Safety in Large Language Models

This repository implements the attribution-driven prompt editing workflow described in our paper *“Editing Prompts to Optimize a Margin of Safety in Large
Language Models.”*

## Repository layout

- `safetymargin/` – core Python package (scenario builders, AOI attribution, HF model wrapper).
- `scripts/` – runnable entry points:
  - `pipeline.py` – end-to-end scenario scoring + LOO/AOI analysis.
  - `plot_*.py`, `figure_*.py`, `analyze_results.py` – figure and table builders.
- `data/` – JSON definitions for the five safety cases from the paper.
- `AAAI2026_MarginofSafety.pdf` – reference manuscript describing the method.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The pipeline expects access to a HuggingFace model checkpoint.  Set
`HUGGINGFACE_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your environment if the
model requires authentication; `scripts/pipeline.py` performs a best-effort
login via `huggingface_hub.login`.

## Running the SafetyMargin pipeline

```bash
python scripts/pipeline.py \
  --case-file data/train/evidence_fixed.json \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --device cuda \
  --output-dir results \
  --save-json
```

The driver follows the procedure from the paper:

1. Load a scenario (`Minimal context` + authored spans).
2. Score aligned vs violating references to obtain the directional safety margin.
3. Run Leave-One-Out (LOO) and Add-One-In (AOI) span interventions.
4. Emit a markdown/text summary and structured JSON artefact.
