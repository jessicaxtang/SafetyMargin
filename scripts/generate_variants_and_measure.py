
import json
import pandas as pd
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetymargin.sguard_content_filter import SGuardContentFilter

UNSAFE_LABELS = {"unsafe", "harmful"}

HANDCRAFTED_PATH = Path("dataset/handcrafted_v2/handcrafted_dataset2.json")
N_GEN = 10  # Number of generations per prompt variant

# --- CONFIG for LLAMA 3.2 1B---
# PER_UNIT_PATH = Path("experiments-local2-mar30/reference_attribution_n100_seed42_llama-3.2-3b/per_unit_rows.csv")
# MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# --- CONFIG for LLAMA 3.2 3B---    
# PER_UNIT_PATH = Path("experiments-local2-mar30/reference_attribution_n100_seed42_llama-3.2-3b/per_unit_rows.csv")
# MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# --- CONFIG for LLAMA 3 8B---
PER_UNIT_PATH = Path("experiments-local2-mar30/reference_attribution_n100_seed42_llama-3-8b/per_unit_rows.csv")
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 

# PER_UNIT_PATH = Path("experiments-local2-mar30/reference_attribution_n100_seed42_qwen-2.5-7b/per_unit_rows.csv")
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "none"
if MODEL_ID == "meta-llama/Llama-3.2-1B-Instruct":
    model_name = "llama-3.2-1b"
elif MODEL_ID == "meta-llama/Llama-3-8B-Instruct":
    model_name = "llama-3-8b"
elif MODEL_ID == "meta-llama/Llama-3.2-3B-Instruct":
    model_name = "llama-3.2-3b"
elif MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct":
    model_name = "qwen-2.5-1.5b"
elif MODEL_ID == "Qwen/Qwen2.5-3B-Instruct":
    model_name = "qwen-2.5-3b"
elif MODEL_ID == "Qwen/Qwen2.5-7B-Instruct":
    model_name = "qwen-2.5-7b"


# --- LOAD DATA ---
with open(HANDCRAFTED_PATH) as f:
    examples = json.load(f)["examples"]
per_unit = pd.read_csv(PER_UNIT_PATH)

if "ground_truth" not in per_unit.columns:
    raise ValueError(
        "Per-unit attribution file must include a 'ground_truth' column to identify unsafe spans."
    )


def remove_unsafe_units(units_df: pd.DataFrame):
    labels = units_df["ground_truth"].fillna("").str.lower()
    unsafe_mask = labels.isin(UNSAFE_LABELS)
    kept_text = units_df.loc[~unsafe_mask, "unit_text"].tolist()
    removed_text = units_df.loc[unsafe_mask, "unit_text"].tolist()
    # If nothing marked unsafe, keep original prompt as-is
    prompt_variant = " ".join(kept_text) if kept_text else ""
    return prompt_variant, removed_text

# --- GENERATION SETUP ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
filter = SGuardContentFilter(device_map=DEVICE)

print("model loaded:", MODEL_ID)

# Suppress pad_token_id warning
warnings.filterwarnings("ignore", message=r"Setting `pad_token_id` to `eos_token_id`.*for open-end generation.")

results = []
total = len(examples) * 2 * N_GEN  # 2 variants per example
with tqdm(total=total, desc="Generating variants") as pbar:
    for idx, ex in enumerate(examples):
        prompt = ex["prompt"]
        units_df = per_unit[per_unit["example_index"] == idx].sort_values("unit_index")
        unsafe_prompt, removed_units = remove_unsafe_units(units_df)

        variants = {
            "original": {"prompt": prompt, "removed_unit": None},
            "remove_unsafe": {
                "prompt": prompt if not removed_units else unsafe_prompt,
                "removed_unit": removed_units if removed_units else None,
            },
        }

        for vtype, config in variants.items():
            vprompt = config["prompt"] or ""
            if not vprompt.strip():
                # Qwen models cannot handle an empty context (happens when all units are unsafe)
                print(
                    f"Skipping variant '{vtype}' for example {idx} because prompt is empty after removing unsafe units."
                )
                results.append({
                    "example_index": idx,
                    "variant": vtype,
                    "prompt": vprompt,
                    "removed_unit": config["removed_unit"],
                    "prompt_unsafe_any": None,
                    "prompt_safety": None,
                    "generation": None,
                    "unsafe_any": None,
                    "unsafe_labels": None,
                    "skip_reason": "empty_prompt_after_removing_unsafe_units",
                })
                pbar.update(N_GEN)
                continue
            for i in range(N_GEN):
                inputs = tokenizer(vprompt, return_tensors="pt").to(DEVICE)
                output = model.generate(**inputs, max_new_tokens=128)
                gen = tokenizer.decode(output[0], skip_special_tokens=True)
                # Margin calculation (reuse margin code if available, else skip)
                # Unsafe rate (per generation)
                prompt_safety = filter.classify(vprompt)
                prompt_unsafe_any = any(l == "unsafe" for l in prompt_safety.labels.values())
                unsafe = filter.classify(vprompt, gen)
                unsafe_any = any(l == "unsafe" for l in unsafe.labels.values())
                results.append({
                    "example_index": idx,
                    "variant": vtype,
                    "prompt": vprompt,
                    "removed_unit": config["removed_unit"],
                    "prompt_unsafe_any": prompt_unsafe_any,
                    "prompt_safety": prompt_safety.labels,
                    "generation": gen,
                    "unsafe_any": unsafe_any,
                    "unsafe_labels": unsafe.labels,
                })
                pbar.update(1)

# Save results

output_path = Path(f"results_generate_var_n_measure/generation_variants_local2mar30_{model_name}.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Saved generation variants and measurements to {output_path}")
