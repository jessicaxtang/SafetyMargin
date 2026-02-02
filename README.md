# Optimizing a Margin of Safety via Prompt Repair in Large Language Models

This repository implements an attribution-drive prompt repair workflow that optimizes for a margin of safety.

## Directory

- `safetymargin/` – lightweight Python package exposing the `HFModel` wrapper
  with the teacher-forcing scoring utilities needed for log-probability
  experiments.
- `scripts/pipeline.py` – implements the prompt repair workflow
- `scripts/test_repair_transfer.py` - tests prompt repair method on the unseen queries
- `dataset/` - test cases in multiple formats (JSON, YAML, CSV)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set your huggingface token in command line: 
```
huggingface-cli login
```


## To run:

```bash
# Legacy JSON format (backward compatible)
python scripts/pipeline.py --case-file dataset/privacy.json --device cuda

# New simplified YAML format (recommended)
python scripts/pipeline.py --case-file dataset/cases/privacy_ssn.yaml --device cuda

# CSV format for bulk datasets
python scripts/pipeline.py --case-file dataset/cases/bulk_cases.csv --case-id privacy_csv_01 --device cuda
```

**See [DATASET_FORMATS.md](DATASET_FORMATS.md) for format documentation and [dataset/README.md](dataset/README.md) for detailed examples.**