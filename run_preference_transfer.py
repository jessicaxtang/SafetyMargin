#!/usr/bin/env python3
"""Run preference-margin transfer experiments on HH-RLHF."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetymargin.models.huggingface_wrapper import HFModel
try:
    from safetymargin.models.vllm_wrapper import VLLMModelWrapper
    HAS_VLLM = True
    VLLM_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    VLLMModelWrapper = None  # type: ignore
    HAS_VLLM = False
    VLLM_IMPORT_ERROR = exc

from safetymargin.preference_transfer import (
    DEFAULT_POLICY_RULES,
    LLMJudgeScorer,
    LengthNormalizedLogProbScorer,
    RewardModelScorer,
    load_hh_rlhf,
    run_leave_one_out_transfer_experiment,
    set_seed,
)


def _resolve_torch_dtype(dtype_name: str):
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_name, None)


def _patch_llama_remote_code_compat() -> None:
    try:
        from transformers.models.llama import modeling_llama

        if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
            modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
    except Exception:
        pass


def build_reward_model_callable(
    model_name: str,
    device: str,
    torch_dtype: str,
    max_length: int,
) -> Callable[[str, str], float]:
    _patch_llama_remote_code_compat()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )
    dtype = _resolve_torch_dtype(torch_dtype)
    if dtype is None and str(device).startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model_kwargs = {
        "trust_remote_code": True,
        "token": hf_token,
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_kwargs,
    )
    reward_model = reward_model.to(device)
    reward_model.eval()

    def _build_text(prompt: str, response: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )
        return f"User: {prompt}\nAssistant: {response}"

    def reward_fn(prompt: str, response: str) -> float:
        inputs = tokenizer(
            _build_text(prompt, response),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = reward_model(**inputs)
            score = float(outputs.logits.squeeze().item())
        return score

    def reward_fn_batch(prompt: str, responses: List[str]) -> List[float]:
        """Batch-score multiple responses for the same prompt in one forward pass."""
        texts = [_build_text(prompt, r) for r in responses]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = reward_model(**inputs)
            scores = outputs.logits.squeeze(-1)
        if scores.dim() == 0:
            return [float(scores.item())]
        return [float(s) for s in scores.tolist()]

    return reward_fn, reward_fn_batch


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preference transfer experiment under distribution shift")

    p.add_argument("--dataset", type=str, default="harmless", choices=["harmless", "helpful", "combined"])
    p.add_argument("--hf-split", type=str, default="train", help="HF split for HH-RLHF (train/test)")
    p.add_argument("--n", type=int, default=500, help="Number of examples to evaluate")
    p.add_argument("--max-length", type=int, default=None, help="Max token length for prompt/chosen/rejected filtering")

    p.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--reference-device",
        type=str,
        default=None,
        help="Device for transformers reference-margin model (defaults to auto policy when using vLLM).",
    )
    p.add_argument("--torch-dtype", type=str, default="auto")
    p.add_argument(
        "--sampling-backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend for response sampling; reference margins still use transformers teacher forcing.",
    )
    p.add_argument(
        "--sampling-model",
        type=str,
        default=None,
        help="Optional model for sampling backend (defaults to --base-model).",
    )
    p.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size when --sampling-backend vllm.",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization target when --sampling-backend vllm.",
    )

    p.add_argument("--k", type=int, default=5, help="Samples per prompt per model")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=256)

    p.add_argument(
        "--scorer",
        type=str,
        default="proxy",
        choices=["proxy", "llm_judge", "reward_fn"],
        help="Scorer mode for generative responses",
    )
    p.add_argument("--judge-model", type=str, default=None, help="Model name for LLM judge mode")
    p.add_argument(
        "--reward-model",
        type=str,
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        help="Model name for reward model mode (--scorer reward_fn)",
    )
    p.add_argument(
        "--reward-device",
        type=str,
        default="cuda",
        help="Device for reward model scoring (e.g., cpu, cuda)",
    )
    p.add_argument(
        "--reward-torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Torch dtype for reward model",
    )
    p.add_argument(
        "--reward-max-length",
        type=int,
        default=4096,
        help="Max token length for reward model inputs",
    )
    p.add_argument(
        "--reference-batch-size",
        type=int,
        default=8,
        help="Batch size for batched teacher-forced reference margin computation",
    )
    p.add_argument(
        "--reference-mode",
        type=str,
        default="dataset_margin",
        choices=["dataset_margin", "self_sampled"],
        help="Reference margin mode: dataset chosen/rejected teacher forcing vs self-sampled buckets.",
    )
    p.add_argument(
        "--reference-samples",
        type=int,
        default=8,
        help="Number of sampled responses per prompt for --reference-mode self_sampled.",
    )
    p.add_argument(
        "--reference-temperature",
        type=float,
        default=None,
        help="Sampling temperature for self-sampled references (defaults to --temperature).",
    )
    p.add_argument(
        "--reference-top-p",
        type=float,
        default=None,
        help="Top-p for self-sampled references (defaults to --top-p).",
    )
    p.add_argument(
        "--reference-max-tokens",
        type=int,
        default=None,
        help="Max new tokens for self-sampled references (defaults to --max-tokens).",
    )
    p.add_argument(
        "--reference-bucket-method",
        type=str,
        default="top_bottom",
        choices=["top_bottom", "median"],
        help="How to split sampled references into preferred/unpreferred buckets.",
    )
    p.add_argument(
        "--reference-bucket-fraction",
        type=float,
        default=0.25,
        help="Top/bottom fraction used for top_bottom bucketing (clipped to [0.05, 0.5]).",
    )
    p.add_argument(
        "--reference-min-bucket-size",
        type=int,
        default=1,
        help="Minimum size of each reference bucket.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="experiments")

    p.add_argument(
        "--intervention-mode",
        type=str,
        default="policy_rules",
        choices=["prompt_units", "policy_rules"],
        help="LOO intervention target: split prompt text vs remove behavior-policy rules",
    )
    p.add_argument(
        "--policy-placement",
        type=str,
        default="prepend",
        choices=["prepend", "append"],
        help="Where to place the behavior policy block relative to dataset prompt",
    )
    p.add_argument(
        "--policy-rule",
        action="append",
        default=None,
        help="Behavior policy rule; repeat for multiple rules. Uses defaults if omitted.",
    )

    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    reference_device = args.reference_device
    if reference_device is None:
        if args.sampling_backend == "vllm" and torch.cuda.is_available():
            if torch.cuda.device_count() > 1 and args.vllm_tensor_parallel_size == 1:
                reference_device = "cuda:1"
            else:
                reference_device = "cpu"
        else:
            reference_device = args.device

    reward_device = args.reward_device
    if args.scorer == "reward_fn" and args.sampling_backend == "vllm" and torch.cuda.is_available():
        if reward_device in {"cuda", "cuda:0"}:
            if torch.cuda.device_count() > 1 and args.vllm_tensor_parallel_size == 1:
                print(
                    "[device-resolve] vLLM uses GPU0; auto-setting reward model device to cuda:1 "
                    "(override with --reward-device if needed)."
                )
                reward_device = "cuda:1"
            else:
                raise RuntimeError(
                    "reward_fn scorer with vLLM cannot place reward model on GPU0 when vLLM is active. "
                    "Allocate at least 2 GPUs and use --reward-device cuda:1, or use --scorer proxy/llm_judge."
                )

    resolved_policy_rules = [rule.strip() for rule in (args.policy_rule or []) if rule and rule.strip()]
    if not resolved_policy_rules:
        resolved_policy_rules = list(DEFAULT_POLICY_RULES)

    dataset = load_hh_rlhf(
        split=args.dataset,
        n_samples=args.n,
        max_length=args.max_length,
        seed=args.seed,
        hf_split=args.hf_split,
    )

    base_model = HFModel(
        model_name=args.base_model,
        auto_load=True,
        device=reference_device,
        torch_dtype=args.torch_dtype,
    )

    sampling_model = base_model
    sampling_model_name = args.sampling_model or args.base_model
    if args.sampling_backend == "vllm":
        if not HAS_VLLM:
            raise RuntimeError(f"vLLM backend requested but unavailable: {VLLM_IMPORT_ERROR}")
        sampling_model = VLLMModelWrapper(  # type: ignore[operator]
            model_name=sampling_model_name,
            auto_load=True,
            max_length=args.max_length or 4096,
            torch_dtype=args.torch_dtype,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )

    if args.scorer == "proxy":
        scorer = LengthNormalizedLogProbScorer(base_model)
    elif args.scorer == "llm_judge":
        if not args.judge_model:
            raise ValueError("--judge-model is required when --scorer llm_judge")
        judge_model = HFModel(
            model_name=args.judge_model,
            auto_load=True,
            device=reference_device,
            torch_dtype=args.torch_dtype,
        )
        scorer = LLMJudgeScorer(judge_model)
    else:
        reward_fn, reward_fn_batch = build_reward_model_callable(
            model_name=args.reward_model,
            device=reward_device,
            torch_dtype=args.reward_torch_dtype,
            max_length=args.reward_max_length,
        )
        scorer = RewardModelScorer(reward_fn, reward_fn_batch=reward_fn_batch)

    dataset_manifest = [
        {
            "index": idx,
            "prompt_text": ex.prompt,
            "chosen_text": ex.chosen,
            "rejected_text": ex.rejected,
            "prompt_sha256": hashlib.sha256(ex.prompt.encode("utf-8")).hexdigest(),
            "chosen_sha256": hashlib.sha256(ex.chosen.encode("utf-8")).hexdigest(),
            "rejected_sha256": hashlib.sha256(ex.rejected.encode("utf-8")).hexdigest(),
            "prompt_chars": len(ex.prompt),
            "chosen_chars": len(ex.chosen),
            "rejected_chars": len(ex.rejected),
        }
        for idx, ex in enumerate(dataset)
    ]

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "intervention_mode": args.intervention_mode,
        "policy_placement": args.policy_placement,
        "policy_rules": resolved_policy_rules,
        "base_model": args.base_model,
        "reference_device": reference_device,
        "sampling_backend": args.sampling_backend,
        "sampling_model": sampling_model_name,
        "scorer_mode": args.scorer,
        "judge_model": args.judge_model,
        "reward_model": args.reward_model,
        "reward_device_resolved": reward_device,
        "dataset": args.dataset,
        "hf_split": args.hf_split,
        "n_examples": len(dataset),
    }

    output_directory = Path(f"{args.output_dir}-{args.dataset}/preference_transfer_n{args.n}_k{args.k}_seed{args.seed}")

    if args.reference_mode == "self_sampled":
        output_directory = Path(f"{args.output_dir}-{args.dataset}/preference_transfer_n{args.n}_k{args.k}_seed{args.seed}_selfsampled")

    results = run_leave_one_out_transfer_experiment(
        model=base_model,
        sampling_model=sampling_model,
        dataset=dataset,
        scorer=scorer,
        k_samples=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
        output_dir=output_directory,
        run_metadata=run_metadata,
        dataset_manifest=dataset_manifest,
        intervention_mode=args.intervention_mode,
        policy_rules=resolved_policy_rules,
        policy_placement=args.policy_placement,
        reference_batch_size=args.reference_batch_size,
        reference_mode=args.reference_mode,
        reference_samples=args.reference_samples,
        reference_temperature=args.reference_temperature,
        reference_top_p=args.reference_top_p,
        reference_max_tokens=args.reference_max_tokens,
        reference_bucket_method=args.reference_bucket_method,
        reference_bucket_fraction=args.reference_bucket_fraction,
        reference_min_bucket_size=args.reference_min_bucket_size,
    )

    rows = results.get("rows", [])
    print("\nTransfer summary")
    print("intervention\tmean_Δref\tmean_Δgen\tpearson_r\tspearman_ρ")
    for row in rows:
        row_name = row.get("edit_id", row.get("intervention", "run"))
        print(
            f"{row_name}\t"
            f"{row['mean_delta_ref']:.6f}\t"
            f"{row['mean_delta_gen']:.6f}\t"
            f"{row['pearson_r']:.6f}\t"
            f"{row['spearman_rho']:.6f}"
        )

    print(f"Results CSV: {results['results_csv']}")
    print(f"Config JSON: {results['config_path']}")
    if "detailed_json_path" in results:
        print(f"Detailed JSON: {results['detailed_json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
