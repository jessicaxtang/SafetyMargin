

# run experiment on handcrafted dataset

## models
- llama-3.2-1b: meta-llama/Llama-3.2-1B-Instruct
- llama-3.2-3b: meta-llama/Llama-3.2-3B-Instruct
- llama-3-8b: meta-llama/Meta-Llama-3-8B-Instruct
- qwen-2.5-1.5b: Qwen/Qwen2.5-1.5B-Instruct
- qwen-2.5-3b: Qwen/Qwen2.5-3B-Instruct
- qwen-2.5-7b: Qwen/Qwen2.5-7B-Instruct

## datasets
- dataset/handcrafted_v1/handcrafted_dataset1.json
- dataset/handcrafted_v2/handcrafted_dataset2.json
- dataset/basic/basic.json

## series of scripts

toy dataset:
```bash
python scripts/reference_margin_attribution.py --local-dataset dataset/basic/basic.json --base-model meta-llama/Llama-3.2-1B-Instruct --intervention-mode prompt_units --prompt-unit-splitter nltk_sentence --output-dir experiments-basic-mar30
```

safety custom dataset:
```bash
python scripts/reference_margin_attribution.py --local-dataset dataset/handcrafted_v2/handcrafted_dataset2.json --base-model meta-llama/Llama-3.2-3B-Instruct --intervention-mode prompt_units --prompt-unit-splitter nltk_sentence --output-dir experiments-local2-mar30
```

python scripts/reference_margin_attribution.py --local-dataset dataset/handcrafted_v2/handcrafted_dataset2.json --base-model Qwen/Qwen2.5-7B-Instruct --intervention-mode prompt_units --prompt-unit-splitter nltk_sentence --output-dir experiments-local2-mar30

plot heatmap/aggregate metrics etc
```bash
python visualization/figure_heatmap.py --model-name llama-3.2-1b
```

visualize span contributions and save as png
```bash
python visualization/example_span_colormap.py --per-unit-path experiments-local2/reference_attribution_n100_seed42_llama-3.2-1b/per_unit_rows.csv --example-index 0
```

test generation variations and rate of harmful outputs

```bash
python scripts/generate_variants_and_measure.py
```

analyze that result^

```bash
python scripts/analyze_generation_variants.py
```

GENERATE VARIANTS AND MEASURE: HH-RLHF

```bash
python scripts/reference_margin_attribution.py --dataset helpful --n 100 --seed 42 --base-model meta-llama/Llama-3.2-3B-Instruct --intervention-mode policy_rules --output-dir experiments-helpful-policy
```

RUN GENERATE AND MEASURE WITH VLLM
```bash
python scripts/generate_variants_and_measure_hh.py     --per-unit-csv experiments-helpful-policy/reference_attribution_n100_seed42_llama-3.2-3b/per_unit_rows.csv     --base-model meta-llama/Llama-3.2-3B-Instruct     --reward-model weqweasdas/hh_rlhf_rm_open_llama_3b     --n-gen 5     --device cuda     --rm-device cuda     --use-vllm     --gpu-memory-utilization 0.55     --output-dir results_transfer_hh
```
