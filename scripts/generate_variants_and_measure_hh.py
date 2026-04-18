"""Transfer correlation experiment for HH-RLHF helpful split (policy-rule LOO).

Two-stage pipeline:

  Stage 1 — compute delta_ref via reference_margin_attribution.py with
  --intervention-mode policy_rules.  Each LOO unit is one rule from
  HELPFUL_POLICY_RULES; removing it produces a reduced prompt whose teacher-
  forced margin vs. the HH-RLHF chosen/rejected pair gives delta_ref.

  Stage 2 — this script.  For each (example, rule) pair read from the Stage 1
  per_unit_rows.csv, generate k responses under the full-policy and
  reduced-policy prompts, score both with a preference reward model, and record
  delta_gen = mean_reward(reduced) - mean_reward(full).  The transfer
  correlation (Pearson r, Spearman rho) between delta_ref and delta_gen is the
  main result.

Intervention design (HELPFUL_POLICY_RULES in preference_transfer.py):
  Using policy-rule LOO rather than dialogue-turn LOO avoids breaking the
  semantic and logical flow of the HH-RLHF dialogues.  Each rule is an
  independent, self-contained instruction whose removal has a clear intended
  direction of effect on the helpfulness reward.

Output files (written to --output-dir/<model-slug>/):
  transfer_results.csv   per-(example, rule) rows with delta_ref, delta_gen,
                         mean rewards, per-example z-scores
  summary.json           run metadata, Pearson r, Spearman rho, p-values

Recommended two-command run:

  # Stage 1: compute delta_ref for policy-rule LOO on helpful split
  python scripts/reference_margin_attribution.py --dataset helpful --n 100 --seed 42 --base-model meta-llama/Llama-3.2-3B-Instruct --intervention-mode policy_rules --output-dir experiments-helpful-policy

      --policy-rule "Answer the user's question as directly and completely as possible."
      --policy-rule "Provide specific details, steps, or examples when they would make your answer more useful."
      --policy-rule "Only decline a request if it would cause clear harm; do not refuse out of excessive caution." 
      --policy-rule "Be honest about what you do not know — do not guess or fabricate information." 
      --policy-rule "Do not add unsolicited warnings, disclaimers, or caveats beyond what is necessary." 
      --policy-rule "Keep your response focused on what the user actually asked; avoid padding or tangents." 

  # Stage 2: generate responses + score with RM, compute transfer correlation
  python scripts/generate_variants_and_measure_hh.py --per-unit-csv experiments-helpful-policy/reference_attribution_n100_seed42_llama-3.2-1b/per_unit_rows.csv --base-model meta-llama/Llama-3.2-1B-Instruct --reward-model weqweasdas/hh_rlhf_rm_open_llama_3b --n-gen 5 --device cuda:0 --rm-device cuda:1 --output-dir results_transfer_hh --max-examples 5
"""

from __future__ import annotations

import os
# Must be set before any CUDA or vLLM import so vLLM's engine core
# subprocess uses 'spawn' rather than 'fork' (fork fails once CUDA is live).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetymargin.models.huggingface_wrapper import HFModel
from safetymargin.models.vllm_wrapper import VLLMModelWrapper
from safetymargin.preference_transfer import set_seed


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class PreferenceRewardModel:
    """Thin wrapper around a HuggingFace sequence-classification reward model.

    Expects the model to output a single scalar logit per sequence
    (num_labels=1), which is standard for RM fine-tunes in the TRL/trlX
    ecosystem (e.g. weqweasdas/hh_rlhf_rm_open_llama_3b).
    """

    def __init__(self, model_id: str, device: str = "cuda", torch_dtype: str = "auto") -> None:
        self.device = device
        dtype = torch.float16 if torch_dtype == "float16" else (
            torch.bfloat16 if torch_dtype == "bfloat16" else "auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {"device_map": device}
        if dtype != "auto":
            load_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, **load_kwargs
        )
        self.model.eval()

    @torch.no_grad()
    def score(self, prompt: str, response: str, max_length: int = 1024) -> float:
        """Return scalar reward for a prompt+response pair.

        The full conversation text (prompt + response) is fed to the RM.
        The separator between prompt and response is omitted when the prompt
        already ends with "Assistant:" — the generated response text is
        appended directly.
        """
        # prompt from per_unit_rows already ends with "\n\nAssistant:"
        # response is the new tokens only (no repeated prompt)
        full_text = prompt + " " + response.strip()
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model(**inputs)
        # logits shape: (1, 1) for num_labels=1
        return float(output.logits[0, 0].item())

    @torch.no_grad()
    def score_batch(
        self, texts: List[Tuple[str, str]], max_length: int = 1024
    ) -> List[float]:
        """Score a batch of (prompt, response) pairs."""
        full_texts = [p + " " + r.strip() for p, r in texts]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model(**inputs)
        return [float(v.item()) for v in output.logits[:, 0]]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _generate_k(
    model: Any,
    prompt: str,
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> List[str]:
    """Generate k responses for a single prompt.

    Uses VLLMModelWrapper.generate_batch (n=k) when available for efficiency,
    falling back to k sequential HFModel.generate calls otherwise.
    """
    if isinstance(model, VLLMModelWrapper):
        results = model.generate_batch(
            [prompt],
            n_per_prompt=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return results[0]  # list of k strings for the single prompt
    # HFModel fallback
    responses: List[str] = []
    for _ in range(k):
        out = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_log_probs=False,
        )
        responses.append(out.generated_text if hasattr(out, "generated_text") else str(out))
    return responses


def _generate_k_batch(
    model: Any,
    prompts: List[str],
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> List[List[str]]:
    """Generate k responses for each prompt in a single batched vLLM call.

    Returns list of len(prompts), each element a list of k strings.
    Falls back to sequential _generate_k calls for HFModel.
    """
    if isinstance(model, VLLMModelWrapper):
        return model.generate_batch(
            prompts,
            n_per_prompt=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    return [_generate_k(model, p, k, temperature, top_p, max_new_tokens) for p in prompts]


# ---------------------------------------------------------------------------
# Per-example z-score standardization
# ---------------------------------------------------------------------------

def _standardize_per_example(
    example_indices: List[int],
    values: List[float],
    min_count: int = 2,
) -> Tuple[List[float], int]:
    z_scores = [float("nan")] * len(values)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for row_idx, ex_idx in enumerate(example_indices):
        grouped[ex_idx].append(row_idx)

    n_standardized = 0
    for indices in grouped.values():
        valid = [i for i in indices if not math.isnan(float(values[i]))]
        if len(valid) < min_count:
            continue
        arr = np.array([float(values[i]) for i in valid])
        std = float(np.std(arr))
        if std <= 1e-12:
            continue
        mean = float(np.mean(arr))
        for pos, i in enumerate(valid):
            z_scores[i] = float((arr[pos] - mean) / std)
        n_standardized += 1
    return z_scores, n_standardized


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _load_per_unit_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HH-RLHF transfer correlation: delta_ref vs delta_gen"
    )
    parser.add_argument(
        "--per-unit-csv",
        required=True,
        help="per_unit_rows.csv from reference_margin_attribution.py run on HH-RLHF helpful split",
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Generative model used to produce responses",
    )
    parser.add_argument(
        "--reward-model",
        default="weqweasdas/hh_rlhf_rm_open_llama_3b",
        help="HuggingFace preference reward model (num_labels=1 sequence classification)",
    )
    parser.add_argument("--n-gen", type=int, default=5, help="Responses to sample per prompt variant")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--rm-max-length", type=int, default=1024, help="Max token length for RM scoring")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for the generative model")
    parser.add_argument("--rm-device", type=str, default=None,
                        help="Device for the reward model (defaults to --device). "
                             "Set to e.g. 'cuda:1' to split across GPUs.")
    parser.add_argument("--torch-dtype", type=str, default="float16",
                        choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Process at most this many examples (useful for quick checks)",
    )
    parser.add_argument("--output-dir", type=str, default="results_transfer_hh")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for generation (batched, much faster). Recommended for sbatch runs.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size (number of GPUs for the generative model).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory utilization fraction (default 0.85).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    warnings.filterwarnings(
        "ignore", message=r"Setting `pad_token_id` to `eos_token_id`.*"
    )

    per_unit_path = Path(args.per_unit_csv)
    if not per_unit_path.exists():
        print(f"ERROR: per_unit_csv not found: {per_unit_path}", file=sys.stderr)
        return 1

    print(f"Loading per-unit rows from {per_unit_path}")
    rows = _load_per_unit_csv(per_unit_path)
    print(f"  Loaded {len(rows)} rows")

    # Group rows by example_index, preserving order
    example_to_rows: Dict[int, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        example_to_rows[int(row["example_index"])].append(i)

    example_indices_sorted = sorted(example_to_rows.keys())
    if args.max_examples is not None:
        example_indices_sorted = example_indices_sorted[: args.max_examples]

    # Load generative model
    if args.use_vllm:
        print(f"Loading generative model (vLLM): {args.base_model}")
        gen_model: Any = VLLMModelWrapper(
            model_name=args.base_model,
            auto_load=True,
            device=args.device,
            torch_dtype=args.torch_dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        print(f"Loading generative model (HF): {args.base_model}")
        gen_model = HFModel(
            model_name=args.base_model,
            auto_load=True,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )

    rm_device = args.rm_device if args.rm_device else args.device
    print(f"Loading reward model: {args.reward_model} (device: {rm_device})")
    rm = PreferenceRewardModel(
        model_id=args.reward_model,
        device=rm_device,
        torch_dtype=args.torch_dtype,
    )

    # Run generation + scoring
    output_rows: List[Dict[str, Any]] = []

    total_examples = len(example_indices_sorted)
    pbar = tqdm(total=total_examples, desc="Examples")

    for example_index in example_indices_sorted:
        row_indices = example_to_rows[example_index]
        example_row_data = [rows[i] for i in row_indices]

        full_prompt = example_row_data[0].get("full_prompt", "").strip()
        if not full_prompt:
            for row in example_row_data:
                output_rows.append({**row, "delta_gen": float("nan"),
                                    "mean_full_reward": float("nan"),
                                    "mean_reduced_reward": float("nan"),
                                    "skip_reason": "empty_full_prompt"})
            pbar.update(1)
            continue

        # Collect all valid reduced prompts for this example
        reduced_prompts = [row.get("reduced_prompt", "").strip() for row in example_row_data]

        # Batch generate k responses for full + all reduced prompts in one vLLM call
        all_prompts = [full_prompt] + reduced_prompts
        all_responses = _generate_k_batch(
            gen_model, all_prompts, args.n_gen,
            args.temperature, args.top_p, args.max_new_tokens,
        )
        full_responses = all_responses[0]
        full_rewards = rm.score_batch(
            [(full_prompt, r) for r in full_responses],
            max_length=args.rm_max_length,
        )
        mean_full_reward = float(np.mean(full_rewards))

        # Score each reduced prompt using the pre-generated batched responses
        for unit_pos, row in enumerate(example_row_data):
            reduced_prompt = reduced_prompts[unit_pos]
            reduced_responses = all_responses[unit_pos + 1]  # offset by 1 (full is index 0)
            out_row: Dict[str, Any] = {**row}

            if not reduced_prompt:
                out_row.update({
                    "mean_full_reward": mean_full_reward,
                    "mean_reduced_reward": float("nan"),
                    "delta_gen": float("nan"),
                    "skip_reason": "empty_reduced_prompt",
                })
                output_rows.append(out_row)
                continue

            reduced_rewards = rm.score_batch(
                [(reduced_prompt, r) for r in reduced_responses],
                max_length=args.rm_max_length,
            )
            mean_reduced_reward = float(np.mean(reduced_rewards))

            # delta_gen = reduced - full  (negative when rule was helpful)
            # Same sign convention as delta_ref = reduced_margin - full_margin
            delta_gen = mean_reduced_reward - mean_full_reward

            out_row.update({
                "mean_full_reward": mean_full_reward,
                "mean_reduced_reward": mean_reduced_reward,
                "delta_gen": delta_gen,
                "full_reward_responses": json.dumps(full_rewards),
                "reduced_reward_responses": json.dumps(reduced_rewards),
                "skip_reason": "",
            })
            output_rows.append(out_row)

        pbar.update(1)

    pbar.close()

    # Per-example z-score standardization for both delta_ref and delta_gen
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    valid_indices = [
        i for i, r in enumerate(output_rows)
        if not math.isnan(_safe_float(r.get("delta_ref")))
        and not math.isnan(_safe_float(r.get("delta_gen")))
    ]

    ex_indices = [int(output_rows[i]["example_index"]) for i in valid_indices]
    delta_ref_vals = [_safe_float(output_rows[i]["delta_ref"]) for i in valid_indices]
    delta_gen_vals = [_safe_float(output_rows[i]["delta_gen"]) for i in valid_indices]

    delta_ref_z, n_ref_z = _standardize_per_example(ex_indices, delta_ref_vals)
    delta_gen_z, n_gen_z = _standardize_per_example(ex_indices, delta_gen_vals)

    # Write z-scores back into output_rows in place
    for list_pos, row_idx in enumerate(valid_indices):
        output_rows[row_idx]["delta_ref_z"] = delta_ref_z[list_pos]
        output_rows[row_idx]["delta_gen_z"] = delta_gen_z[list_pos]

    # Compute correlation on paired (delta_ref_z, delta_gen_z) with both finite
    paired_ref = [rz for rz, gz in zip(delta_ref_z, delta_gen_z)
                  if not math.isnan(rz) and not math.isnan(gz)]
    paired_gen = [gz for rz, gz in zip(delta_ref_z, delta_gen_z)
                  if not math.isnan(rz) and not math.isnan(gz)]

    pearson_r = float("nan")
    pearson_p = float("nan")
    spearman_rho = float("nan")
    spearman_p = float("nan")
    n_pairs = len(paired_ref)

    if n_pairs >= 3:
        r_val, p_val = scipy_stats.pearsonr(paired_ref, paired_gen)
        pearson_r, pearson_p = float(r_val), float(p_val)
        rho_val, sp_val = scipy_stats.spearmanr(paired_ref, paired_gen)
        spearman_rho, spearman_p = float(rho_val), float(sp_val)

    # Derive model name slug for output dir
    model_slug = args.base_model.split("/")[-1].lower().replace("_", "-")

    out_dir = Path(args.output_dir) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "transfer_results.csv"
    _write_csv(csv_path, output_rows)

    # Save summary
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "n_examples_processed": len(example_indices_sorted),
        "n_unit_rows_total": len(output_rows),
        "n_unit_rows_valid": len(valid_indices),
        "n_pairs_for_correlation": n_pairs,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "delta_gen_mean": float(np.mean(delta_gen_vals)) if delta_gen_vals else float("nan"),
        "delta_gen_std": float(np.std(delta_gen_vals)) if delta_gen_vals else float("nan"),
        "n_ref_z_examples": n_ref_z,
        "n_gen_z_examples": n_gen_z,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nTransfer Correlation Summary")
    print(f"  Examples processed : {summary['n_examples_processed']}")
    print(f"  Valid pairs        : {n_pairs}")
    print(f"  Pearson r          : {pearson_r:.4f}  (p={pearson_p:.4f})")
    print(f"  Spearman rho       : {spearman_rho:.4f}  (p={spearman_p:.4f})")
    print(f"  Results CSV        : {csv_path}")
    print(f"  Summary JSON       : {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
