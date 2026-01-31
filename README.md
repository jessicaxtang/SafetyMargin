# Optimizing a Margin of Safety via Prompt Repair in Large Language Models

This repository implements an attribution-drive prompt repair workflow that optimizes for a margin of safety.

## Directory

- `safetymargin/` – lightweight Python package exposing the `HFModel` wrapper
  with the teacher-forcing scoring utilities needed for log-probability
  experiments.
- `scripts/pipeline.py` – implements the prompt repair workflow
- `scripts/test_repair_transfer.py` - tests prompt repair method on the unseen queries
- `data/` - contains prompts and spans from 5 safety cases

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```