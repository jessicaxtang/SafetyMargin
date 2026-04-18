"""Preference margin transfer experiments under distribution shift.

This module evaluates whether improvements in teacher-forced reference margins
transfer to improvements under sampled generations.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from safetymargin.utils import extract_generated_text


@dataclass
class Example:
    """Pairwise preference example."""

    prompt: str
    chosen: str    
    rejected: str


DEFAULT_POLICY_RULES: List[str] = [
    "Answer the user's request directly and helpfully.",
    "Be accurate and avoid unsupported claims.",
    "If information is uncertain, say so clearly.",
    "Avoid harmful or unsafe assistance.",
    "Respect privacy and do not infer sensitive personal information."
]

# Expected tags for policy rule validation (for S1 attribution validity scoring)
DEFAULT_POLICY_RULE_TAGS: List[str] = [
    "helpful",    # "Answer the user's request directly and helpfully."
    "helpful",    # "Be accurate and avoid unsupported claims."
    "neutral",    # "If information is uncertain, say so clearly."
    "harmful",    # "Avoid harmful or unsafe assistance."
    "neutral",    # "Respect privacy and do not infer sensitive personal information."
]

# Policy rules designed for LOO attribution on the HH-RLHF helpful split.
#
# The rules are chosen to span a range of effects on the helpfulness reward signal:
#   - Rules 1-2: core helpfulness drivers; removing them should substantially reduce
#     both the reference margin and the mean reward (strong positive attribution)
#   - Rule 3: anti-over-refusal; removing it may allow unnecessary refusals, slightly
#     reducing reward on the helpful split (moderate positive attribution)
#   - Rule 4: honesty / calibration; removing it permits confident but wrong answers,
#     which a well-calibrated RM should penalise (moderate attribution, direction varies)
#   - Rules 5-6: constraining rules that can reduce response quality by adding
#     unnecessary hedges or limiting scope; removing them may slightly increase reward
#     (near-zero or negative attribution)
#
# This spread creates variance in delta_ref and delta_gen, which is necessary for a
# meaningful transfer correlation.
HELPFUL_POLICY_RULES: List[str] = [
    "Answer the user's question as directly and completely as possible.",
    "Provide specific details, steps, or examples when they would make your answer more useful.",
    "Only decline a request if it would cause clear harm; do not refuse out of excessive caution.",
    "Be honest about what you do not know — do not guess or fabricate information.",
    "Do not add unsolicited warnings, disclaimers, or caveats beyond what is necessary.",
    "Keep your response focused on what the user actually asked; avoid padding or tangents.",
]

HELPFUL_POLICY_RULE_TAGS: List[str] = [
    "helpful",    # "Answer the user's question as directly and completely as possible."
    "helpful",    # "Provide specific details, steps, or examples..."
    "helpful",    # "Only decline a request if it would cause clear harm..."
    "neutral",    # "Be honest about what you do not know..."
    "constraining",  # "Do not add unsolicited warnings or disclaimers..."
    "constraining",  # "Keep your response focused..."
]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def _tokenize_whitespace(text: str) -> List[str]:
    return re.findall(r"\S+", (text or "").strip())


def _estimate_length(text: str, tokenizer: Any = None) -> int:
    if tokenizer is not None:
        try:
            enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            return int(enc["input_ids"].shape[1])
        except Exception:
            pass
    return len(_tokenize_whitespace(text))


def _split_hh_dialogue(transcript: str) -> Tuple[str, str]:
    """Split HH transcript into prompt and final assistant response.

    Works on strings containing repeated "Human:" / "Assistant:" turns.
    """
    text = (transcript or "").strip()
    if not text:
        return "", ""

    markers = ["\n\nAssistant:", "\nAssistant:", "Assistant:"]
    idx = -1
    marker = ""
    for candidate in markers:
        candidate_idx = text.rfind(candidate)
        if candidate_idx > idx:
            idx = candidate_idx
            marker = candidate

    if idx < 0:
        return text, ""

    prompt = text[: idx + len(marker)].strip()
    response = text[idx + len(marker) :].strip()
    return prompt, response


def _extract_pairwise_example(chosen_raw: str, rejected_raw: str) -> Optional[Example]:
    """Extract (prompt, chosen, rejected) from HH chosen/rejected transcripts."""
    chosen_prompt, chosen_resp = _split_hh_dialogue(chosen_raw)
    rejected_prompt, rejected_resp = _split_hh_dialogue(rejected_raw)

    if chosen_prompt == rejected_prompt and chosen_resp and rejected_resp:
        return Example(prompt=chosen_prompt, chosen=chosen_resp, rejected=rejected_resp)

    common = ""
    for char_a, char_b in zip(chosen_raw, rejected_raw):
        if char_a != char_b:
            break
        common += char_a

    boundary = max(common.rfind("\n\nAssistant:"), common.rfind("\nAssistant:"), common.rfind("Assistant:"))
    if boundary < 0:
        prompt = chosen_prompt or rejected_prompt
        if not prompt:
            return None
        return Example(prompt=prompt, chosen=chosen_resp, rejected=rejected_resp)

    marker_len = len("Assistant:")
    prompt = common[: boundary + marker_len].strip()
    if not prompt:
        return None

    chosen_suffix = chosen_raw[boundary + marker_len :].strip()
    rejected_suffix = rejected_raw[boundary + marker_len :].strip()
    if not chosen_suffix or not rejected_suffix:
        return None

    return Example(prompt=prompt, chosen=chosen_suffix, rejected=rejected_suffix)


def load_hh_rlhf(
    split: str,
    n_samples: int,
    max_length: Optional[int] = None,
    seed: int = 0,
    hf_split: str = "train",
) -> List[Example]:
    """Load HH-RLHF examples as pairwise preference tuples.

    Args:
        split: One of {"harmless", "helpful", "combined"}.
        n_samples: Number of examples to return.
        max_length: Optional max token length per field (prompt/chosen/rejected).
        seed: Shuffle seed.
        hf_split: HF split (usually train/test).
    """
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required for HH-RLHF loading") from exc

    split_key = (split or "combined").strip().lower()
    if split_key not in {"harmless", "helpful", "combined"}:
        raise ValueError("split must be one of: harmless, helpful, combined")

    config_map: Dict[str, List[str]] = {
        "harmless": ["harmless-base", "harmless"],
        "helpful": ["helpful-base", "helpful", "helpful-online", "helpful-rejection-sampled"],
        "combined": ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"],
    }

    def _try_load_subset(subset_name: str):
        """Try multiple loading patterns because HH-RLHF packaging varies by datasets version."""
        load_attempts = [
            lambda: load_dataset("Anthropic/hh-rlhf", subset_name, split=hf_split),
            lambda: load_dataset("Anthropic/hh-rlhf", data_dir=subset_name, split=hf_split),
            lambda: load_dataset("Anthropic/hh-rlhf", split=f"{subset_name}/{hf_split}"),
            lambda: load_dataset(
                "json",
                data_files={
                    hf_split: f"hf://datasets/Anthropic/hh-rlhf/{subset_name}/{hf_split}.jsonl.gz",
                },
                split=hf_split,
            ),
        ]
        for attempt in load_attempts:
            try:
                loaded_ds = attempt()
                if loaded_ds is not None:
                    return loaded_ds
            except Exception:
                continue
        return None

    datasets_to_merge = []
    for cfg in config_map[split_key]:
        loaded = _try_load_subset(cfg)
        if loaded is not None:
            datasets_to_merge.append(loaded)

    if not datasets_to_merge:
        try:
            fallback_ds = load_dataset("Anthropic/hh-rlhf", split=hf_split)
        except Exception as exc:
            raise RuntimeError("Failed to load Anthropic/hh-rlhf with expected configurations") from exc

        # Some HH-RLHF packaging variants expose a unified split with a source field.
        if split_key == "combined":
            datasets_to_merge = [fallback_ds]
        else:
            source_col = None
            for candidate in ("source", "subset", "data_source", "split"):
                if hasattr(fallback_ds, "column_names") and candidate in fallback_ds.column_names:
                    source_col = candidate
                    break

            if source_col is not None:
                target = "harmless" if split_key == "harmless" else "helpful"
                filtered = fallback_ds.filter(
                    lambda row: target in str(row.get(source_col, "")).lower(),
                )
                datasets_to_merge = [filtered]
            else:
                if split_key == "harmless":
                    raise RuntimeError(
                        "Unable to isolate harmless subset from this HH-RLHF packaging; "
                        "try --dataset combined or upgrade datasets package."
                    )
                datasets_to_merge = [fallback_ds]

    examples: List[Example] = []
    rng = random.Random(seed)

    rows: List[Dict[str, Any]] = []
    for ds in datasets_to_merge:
        ds_shuf = ds.shuffle(seed=seed)
        rows.extend(list(ds_shuf))
    rng.shuffle(rows)

    for row in rows:
        chosen_raw = str(row.get("chosen") or "").strip()
        rejected_raw = str(row.get("rejected") or "").strip()
        if not chosen_raw or not rejected_raw:
            continue
        item = _extract_pairwise_example(chosen_raw, rejected_raw)
        if item is None:
            continue
        if max_length is not None:
            if (
                _estimate_length(item.prompt) > max_length
                or _estimate_length(item.chosen) > max_length
                or _estimate_length(item.rejected) > max_length
            ):
                continue
        examples.append(item)
        if len(examples) >= n_samples:
            break

    if len(examples) < n_samples:
        raise RuntimeError(f"Requested {n_samples} examples, but only loaded {len(examples)} after filtering")
    return examples


def _teacher_forced_sum_logprob(model: Any, prompt: str, target: str) -> float:
    if not hasattr(model, "teacher_forcing_forward"):
        raise TypeError("Model must expose teacher_forcing_forward(prompt_text, target_text)")
    result = model.teacher_forcing_forward(prompt_text=prompt, target_text=target, require_grad=False)
    return float(result["sum_log_prob"])


def compute_reference_margin(model: Any, prompt: str, chosen: str, rejected: str) -> float:
    """Compute teacher-forced preference margin M(x)."""
    chosen_lp = _teacher_forced_sum_logprob(model, prompt, chosen)
    rejected_lp = _teacher_forced_sum_logprob(model, prompt, rejected)
    return chosen_lp - rejected_lp


def compute_scorer_reference_margin(scorer: "PreferenceScorer", prompt: str, chosen: str, rejected: str) -> float:
    scores = scorer.score_batch(prompt, [chosen, rejected])
    return scores[0] - scores[1]


def compute_reference_margins_batched(
    model: Any,
    prompts: Sequence[str],
    chosen: str,
    rejected: str,
    batch_size: int = 8,
) -> List[float]:
    """Compute reference margins for many prompts with batched teacher forcing when available."""
    if len(prompts) == 0:
        return []
    if not hasattr(model, "teacher_forcing_forward_batch"):
        return [compute_reference_margin(model, prompt, chosen, rejected) for prompt in prompts]

    margins: List[float] = []
    step = max(int(batch_size), 1)
    start = 0
    while start < len(prompts):
        chunk = list(prompts[start : start + step])
        try:
            chosen_results = model.teacher_forcing_forward_batch(chunk, [chosen] * len(chunk), require_grad=False)
            rejected_results = model.teacher_forcing_forward_batch(chunk, [rejected] * len(chunk), require_grad=False)
        except Exception as exc:
            is_oom = False
            if HAS_TORCH and isinstance(exc, torch.OutOfMemoryError):
                is_oom = True
            elif "out of memory" in str(exc).lower():
                is_oom = True

            if not is_oom:
                raise
            if step <= 1:
                raise RuntimeError(
                    "Out of memory during teacher-forced reference margin computation at batch_size=1. "
                    "If using vLLM sampling, run transformers reference scoring on another device "
                    "(e.g., --reference-device cuda:1) or CPU (--reference-device cpu), and/or reduce model size."
                ) from exc

            step = max(1, step // 2)
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        for chosen_result, rejected_result in zip(chosen_results, rejected_results):
            margins.append(float(chosen_result["sum_log_prob"]) - float(rejected_result["sum_log_prob"]))
        start += len(chunk)
    return margins


def sample_responses(
    model: Any,
    prompt: str,
    k: int,
    temperature: float,
    max_tokens: int,
    top_p: float = 0.95,
) -> List[str]:
    """Sample K responses from a model."""
    if not hasattr(model, "generate"):
        raise TypeError("Model must expose generate()")

    outputs: List[str] = []
    for _ in range(max(k, 0)):
        out = model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            return_log_probs=False,
        )
        text = getattr(out, "generated_text", None)
        if not text:
            text = extract_generated_text(out, model, prompt)
        outputs.append((text or "").strip())
    return outputs


def _split_scores_into_buckets(
    scores: Sequence[float],
    method: str = "top_bottom",
    bucket_fraction: float = 0.5,
    min_bucket_size: int = 1,
) -> Tuple[List[int], List[int]]:
    n = len(scores)
    if n < 2:
        return [], []

    method_key = (method or "top_bottom").strip().lower()
    indexed = sorted(enumerate([float(s) for s in scores]), key=lambda item: item[1], reverse=True)
    min_size = max(int(min_bucket_size), 1)

    if method_key == "median":
        values = np.asarray([float(s) for s in scores], dtype=np.float64)
        med = float(np.median(values))
        pref = [idx for idx, value in enumerate(values.tolist()) if value >= med]
        unpref = [idx for idx, value in enumerate(values.tolist()) if value < med]
        if len(pref) >= min_size and len(unpref) >= min_size:
            return pref, unpref

    frac = min(max(float(bucket_fraction), 0.05), 0.5)
    size = max(min_size, int(math.floor(n * frac)))
    max_allowed = max((n - 1) // 2, 1)
    size = min(size, max_allowed)
    pref_indices = [idx for idx, _ in indexed[:size]]
    unpref_indices = [idx for idx, _ in indexed[-size:]]
    return pref_indices, unpref_indices


def _compute_self_sampled_reference_margin(
    prompt: str,
    responses: Sequence[str],
    scorer: "PreferenceScorer",
    bucket_method: str,
    bucket_fraction: float,
    min_bucket_size: int,
) -> Dict[str, Any]:
    kept_responses: List[str] = list(responses)

    if len(kept_responses) < 2:
        return {
            "ref_margin": float("nan"),
            "responses_all": list(responses),
            "responses_used": kept_responses,
            "pref_count": 0,
            "unpref_count": 0,
            "pref_responses": [],
            "unpref_responses": [],
            "pref_scores": [],
            "unpref_scores": [],
            "bucket_method": bucket_method,
            "bucket_fraction": float(bucket_fraction),
            "scores": [],
        }

    scores = scorer.score_batch(prompt, kept_responses)
    pref_indices, unpref_indices = _split_scores_into_buckets(
        scores=scores,
        method=bucket_method,
        bucket_fraction=bucket_fraction,
        min_bucket_size=min_bucket_size,
    )
    if not pref_indices or not unpref_indices:
        return {
            "ref_margin": float("nan"),
            "responses_all": list(responses),
            "responses_used": kept_responses,
            "pref_count": 0,
            "unpref_count": 0,
            "pref_responses": [],
            "unpref_responses": [],
            "pref_scores": [],
            "unpref_scores": [],
            "bucket_method": bucket_method,
            "bucket_fraction": float(bucket_fraction),
            "scores": [float(s) for s in scores],
        }

    pref_scores = [float(scores[idx]) for idx in pref_indices]
    unpref_scores = [float(scores[idx]) for idx in unpref_indices]
    pref_responses = [kept_responses[idx] for idx in pref_indices]
    unpref_responses = [kept_responses[idx] for idx in unpref_indices]
    ref_margin = float(np.mean(pref_scores) - np.mean(unpref_scores))
    return {
        "ref_margin": ref_margin,
        "responses_all": list(responses),
        "responses_used": kept_responses,
        "pref_count": int(len(pref_scores)),
        "unpref_count": int(len(unpref_scores)),
        "pref_responses": pref_responses,
        "unpref_responses": unpref_responses,
        "pref_scores": pref_scores,
        "unpref_scores": unpref_scores,
        "bucket_method": bucket_method,
        "bucket_fraction": float(bucket_fraction),
        "scores": [float(s) for s in scores],
    }


def _compute_shared_pool_marginals(
    full_prompt: str,
    reduced_prompt: str,
    shared_responses: Sequence[str],
    scorer: "PreferenceScorer",
    bucket_method: str,
    bucket_fraction: float,
    min_bucket_size: int,
) -> Dict[str, Dict[str, Any]]:
    """Compute both full and reduced reference margins from a shared candidate pool.

    Scores the same responses under both the full and reduced prompts, then buckets
    and computes margins for each. This isolates scoring changes from sampling variance.

    Args:
        full_prompt: Prompt with all policy rules prepended
        reduced_prompt: Prompt with one rule removed
        shared_responses: Same responses to be scored under both prompts
        scorer: Scorer to use for evaluating responses
        bucket_method: "top_bottom" or "median"
        bucket_fraction: Bucket size as fraction of total responses
        min_bucket_size: Minimum size per bucket

    Returns:
        Dict with "full" and "reduced" keys, each containing marginal information
    """
    kept_responses = list(shared_responses)

    if len(kept_responses) < 2:
        empty_result = {
            "ref_margin": float("nan"),
            "responses_all": list(shared_responses),
            "responses_used": kept_responses,
            "pref_count": 0,
            "unpref_count": 0,
            "pref_responses": [],
            "unpref_responses": [],
            "pref_scores": [],
            "unpref_scores": [],
            "bucket_method": bucket_method,
            "bucket_fraction": float(bucket_fraction),
            "scores": [],
        }
        return {"full": empty_result, "reduced": empty_result}

    # Score under full prompt
    full_scores = scorer.score_batch(full_prompt, kept_responses)
    pref_indices_full, unpref_indices_full = _split_scores_into_buckets(
        scores=full_scores,
        method=bucket_method,
        bucket_fraction=bucket_fraction,
        min_bucket_size=min_bucket_size,
    )

    # Score under reduced prompt
    reduced_scores = scorer.score_batch(reduced_prompt, kept_responses)
    pref_indices_reduced, unpref_indices_reduced = _split_scores_into_buckets(
        scores=reduced_scores,
        method=bucket_method,
        bucket_fraction=bucket_fraction,
        min_bucket_size=min_bucket_size,
    )

    def make_marginal_result(
        scores: List[float],
        pref_indices: List[int],
        unpref_indices: List[int],
        prompt_context: str,
    ) -> Dict[str, Any]:
        if not pref_indices or not unpref_indices:
            return {
                "ref_margin": float("nan"),
                "responses_all": list(shared_responses),
                "responses_used": kept_responses,
                "pref_count": 0,
                "unpref_count": 0,
                "pref_responses": [],
                "unpref_responses": [],
                "pref_scores": [],
                "unpref_scores": [],
                "bucket_method": bucket_method,
                "bucket_fraction": float(bucket_fraction),
                "scores": [float(s) for s in scores],
            }

        pref_scores = [float(scores[idx]) for idx in pref_indices]
        unpref_scores = [float(scores[idx]) for idx in unpref_indices]
        pref_responses = [kept_responses[idx] for idx in pref_indices]
        unpref_responses = [kept_responses[idx] for idx in unpref_indices]
        ref_margin = float(np.mean(pref_scores) - np.mean(unpref_scores))
        return {
            "ref_margin": ref_margin,
            "responses_all": list(shared_responses),
            "responses_used": kept_responses,
            "pref_count": int(len(pref_scores)),
            "unpref_count": int(len(unpref_scores)),
            "pref_responses": pref_responses,
            "unpref_responses": unpref_responses,
            "pref_scores": pref_scores,
            "unpref_scores": unpref_scores,
            "bucket_method": bucket_method,
            "bucket_fraction": float(bucket_fraction),
            "scores": [float(s) for s in scores],
        }

    return {
        "full": make_marginal_result(full_scores, pref_indices_full, unpref_indices_full, "full"),
        "reduced": make_marginal_result(reduced_scores, pref_indices_reduced, unpref_indices_reduced, "reduced"),
    }


class PreferenceScorer(ABC):
    """Interface for scoring preference quality of generated responses."""

    @abstractmethod
    def score(self, prompt: str, response: str) -> float:
        raise NotImplementedError

    def score_batch(self, prompt: str, responses: List[str]) -> List[float]:
        """Score a batch of responses for the same prompt.

        Override in subclasses for efficient batched computation.
        Falls back to individual score() calls by default.
        """
        return [self.score(prompt, r) for r in responses]


class RewardModelScorer(PreferenceScorer):
    """Adapter for callable reward-model scorers.

    Expects a callable with signature reward_fn(prompt, response) -> float.
    Optionally accepts reward_fn_batch(prompt, responses) -> List[float] for
    efficient batched scoring.
    """

    def __init__(
        self,
        reward_fn: Callable[[str, str], float],
        reward_fn_batch: Optional[Callable[[str, List[str]], List[float]]] = None,
    ):
        self.reward_fn = reward_fn
        self.reward_fn_batch = reward_fn_batch

    def score(self, prompt: str, response: str) -> float:
        return float(self.reward_fn(prompt, response))

    def score_batch(self, prompt: str, responses: List[str]) -> List[float]:
        if not responses:
            return []
        if self.reward_fn_batch is not None:
            return [float(s) for s in self.reward_fn_batch(prompt, responses)]
        return [float(self.reward_fn(prompt, r)) for r in responses]


class LLMJudgeScorer(PreferenceScorer):
    """LLM-judge scorer that asks a judge model for a numeric rating."""

    def __init__(
        self,
        judge_model: Any,
        rubric: Optional[str] = None,
        score_min: float = 0.0,
        score_max: float = 10.0,
    ):
        self.judge_model = judge_model
        self.rubric = rubric or (
            "Rate the assistant response quality for this prompt from 0 to 10. "
            "Respond with only a number."
        )
        self.score_min = score_min
        self.score_max = score_max

    def _build_judge_prompt(self, prompt: str, response: str) -> str:
        return (
            f"{self.rubric}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Response:\n{response}\n\n"
            "Score:"
        )

    def score(self, prompt: str, response: str) -> float:
        judge_prompt = self._build_judge_prompt(prompt, response)
        out = self.judge_model.generate(
            judge_prompt,
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            return_log_probs=False,
        )
        raw_text = getattr(out, "generated_text", None) or extract_generated_text(out, self.judge_model, judge_prompt) or ""
        match = re.search(r"-?\d+(?:\.\d+)?", raw_text)
        if not match:
            return self.score_min
        value = float(match.group(0))
        return float(min(self.score_max, max(self.score_min, value)))


class LengthNormalizedLogProbScorer(PreferenceScorer):
    """Proxy scorer using length-normalized teacher-forced log-probability."""

    def __init__(self, model: Any):
        self.model = model

    def score(self, prompt: str, response: str) -> float:
        if not hasattr(self.model, "teacher_forcing_forward"):
            raise TypeError("Model must expose teacher_forcing_forward() for logprob scoring")
        result = self.model.teacher_forcing_forward(prompt_text=prompt, target_text=response, require_grad=False)
        token_count = 1
        target_log_probs = result.get("target_log_probs")
        if target_log_probs is not None:
            try:
                token_count = max(int(target_log_probs.numel()), 1)
            except Exception:
                token_count = max(len(target_log_probs), 1)
        return float(result["sum_log_prob"]) / token_count


def compute_transfer_correlation(delta_ref_list: Sequence[float], delta_gen_list: Sequence[float]) -> Dict[str, float]:
    """Compute transfer correlation metrics between Δref and Δgen."""
    if len(delta_ref_list) != len(delta_gen_list):
        raise ValueError("delta_ref_list and delta_gen_list must have same length")

    pairs = [
        (float(a), float(b))
        for a, b in zip(delta_ref_list, delta_gen_list)
        if not (math.isnan(float(a)) or math.isnan(float(b)))
    ]
    if len(pairs) < 2:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": float(len(pairs)),
        }

    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": float(len(pairs)),
        }

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n": float(len(pairs)),
    }


def standardize_deltas_per_example(per_unit_rows: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float], Dict[str, float]]:
    example_indices = [int(row.get("example_index", -1)) for row in per_unit_rows]
    delta_ref_values = [float(row.get("delta_ref", float("nan"))) for row in per_unit_rows]
    delta_gen_values = [float(row.get("delta_gen", float("nan"))) for row in per_unit_rows]
    return standardize_pair_lists_per_example(example_indices, delta_ref_values, delta_gen_values)


def standardize_pair_lists_per_example(
    example_indices: Sequence[int],
    ref_values: Sequence[float],
    gen_values: Sequence[float],
) -> Tuple[List[float], List[float], Dict[str, float]]:
    if len(example_indices) != len(ref_values) or len(ref_values) != len(gen_values):
        raise ValueError("example_indices, ref_values, and gen_values must have the same length")

    z_ref = [float("nan")] * len(ref_values)
    z_gen = [float("nan")] * len(gen_values)
    grouped_indices: Dict[int, List[int]] = {}

    for row_idx, example_index in enumerate(example_indices):
        grouped_indices.setdefault(int(example_index), []).append(row_idx)

    standardized_examples = 0
    for indices in grouped_indices.values():
        valid_indices = [
            idx
            for idx in indices
            if not (math.isnan(float(ref_values[idx])) or math.isnan(float(gen_values[idx])))
        ]
        if len(valid_indices) < 2:
            continue

        ref_arr = np.asarray([float(ref_values[idx]) for idx in valid_indices], dtype=np.float64)
        gen_arr = np.asarray([float(gen_values[idx]) for idx in valid_indices], dtype=np.float64)
        ref_std = float(np.std(ref_arr))
        gen_std = float(np.std(gen_arr))
        if ref_std <= 1e-12 or gen_std <= 1e-12:
            continue

        ref_mean = float(np.mean(ref_arr))
        gen_mean = float(np.mean(gen_arr))
        for pos, idx in enumerate(valid_indices):
            z_ref[idx] = float((ref_arr[pos] - ref_mean) / ref_std)
            z_gen[idx] = float((gen_arr[pos] - gen_mean) / gen_std)
        standardized_examples += 1

    return z_ref, z_gen, {"n_examples_standardized": float(standardized_examples)}


def _legacy_standardize_deltas_per_example(per_unit_rows: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float], Dict[str, float]]:
    z_ref = [float("nan")] * len(per_unit_rows)
    z_gen = [float("nan")] * len(per_unit_rows)
    grouped_indices: Dict[int, List[int]] = {}

    for row_idx, row in enumerate(per_unit_rows):
        example_index = int(row.get("example_index", -1))
        grouped_indices.setdefault(example_index, []).append(row_idx)

    standardized_examples = 0
    for indices in grouped_indices.values():
        valid_indices = [
            idx
            for idx in indices
            if not (
                math.isnan(float(per_unit_rows[idx].get("delta_ref", float("nan"))))
                or math.isnan(float(per_unit_rows[idx].get("delta_gen", float("nan"))))
            )
        ]
        if len(valid_indices) < 2:
            continue

        ref_values = np.asarray([float(per_unit_rows[idx]["delta_ref"]) for idx in valid_indices], dtype=np.float64)
        gen_values = np.asarray([float(per_unit_rows[idx]["delta_gen"]) for idx in valid_indices], dtype=np.float64)
        ref_std = float(np.std(ref_values))
        gen_std = float(np.std(gen_values))
        if ref_std <= 1e-12 or gen_std <= 1e-12:
            continue

        ref_mean = float(np.mean(ref_values))
        gen_mean = float(np.mean(gen_values))
        for pos, idx in enumerate(valid_indices):
            z_ref[idx] = float((ref_values[pos] - ref_mean) / ref_std)
            z_gen[idx] = float((gen_values[pos] - gen_mean) / gen_std)
        standardized_examples += 1

    return z_ref, z_gen, {"n_examples_standardized": float(standardized_examples)}


def save_transfer_scatter(
    delta_ref_list: Sequence[float],
    delta_gen_list: Sequence[float],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Save Δref-vs-Δgen scatter with linear fit line."""
    if not HAS_MATPLOTLIB:
        return

    pairs = [
        (float(a), float(b))
        for a, b in zip(delta_ref_list, delta_gen_list)
        if not (math.isnan(float(a)) or math.isnan(float(b)))
    ]
    if len(pairs) < 2:
        return

    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if x.size < 2:
        return

    # Avoid unstable regression fit for constant or near-constant inputs.
    can_fit = bool(np.std(x) > 1e-12 and np.std(y) > 1e-12)

    fig, ax = plt.subplots(figsize=(7, 5))  # type: ignore[arg-type]
    ax.scatter(x, y, alpha=0.7)

    if can_fit:
        try:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linestyle="--")
        except Exception:
            pass

    ax.set_xlabel("Δref")
    ax.set_ylabel("Δgen")
    ax.set_title(title or "Preference Transfer: Δref vs Δgen")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def split_prompt_into_loo_units(prompt: str) -> List[str]:
    """Split prompt text into removable LOO units."""
    text = (prompt or "").strip()
    if not text:
        return []

    units = [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk and chunk.strip()]
    if len(units) >= 2:
        return units

    sentence_units = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk and chunk.strip()]
    if len(sentence_units) >= 2:
        return sentence_units

    return []


def _remove_loo_unit(units: Sequence[str], remove_index: int) -> str:
    kept = [unit for idx, unit in enumerate(units) if idx != remove_index]
    return "\n\n".join(kept).strip()


def _format_behavior_policy_block(policy_rules: Sequence[str]) -> str:
    rules = [str(rule).strip() for rule in policy_rules if str(rule).strip()]
    numbered_rules = "\n".join([f"{idx + 1}. {rule}" for idx, rule in enumerate(rules)])
    return f"[BEHAVIOR POLICY BLOCK]\n\n{numbered_rules}".strip()


def build_prompt_with_behavior_policy(
    dataset_prompt: str,
    policy_rules: Sequence[str],
    placement: str = "prepend",
) -> str:
    policy_block = _format_behavior_policy_block(policy_rules)
    task_block = f"[DATASET PROMPT]\n\n{(dataset_prompt or '').strip()}".strip()
    placement_key = (placement or "prepend").strip().lower()
    if placement_key == "append":
        return f"{task_block}\n\n{policy_block}".strip()
    return f"{policy_block}\n\n{task_block}".strip()


def build_leave_one_out_prompt_variants(
    prompt: str,
    intervention_mode: str = "prompt_units",
    policy_rules: Optional[Sequence[str]] = None,
    policy_placement: str = "prepend",
) -> Tuple[str, List[Tuple[int, str, str]]]:
    mode = (intervention_mode or "prompt_units").strip().lower()
    if mode == "policy_rules":
        rules = [str(rule).strip() for rule in (policy_rules or DEFAULT_POLICY_RULES) if str(rule).strip()]
        if len(rules) < 2:
            return "", []

        full_prompt = build_prompt_with_behavior_policy(
            dataset_prompt=prompt,
            policy_rules=rules,
            placement=policy_placement,
        )
        variants: List[Tuple[int, str, str]] = []
        for idx, rule in enumerate(rules):
            reduced_rules = [candidate for rule_idx, candidate in enumerate(rules) if rule_idx != idx]
            reduced_prompt = build_prompt_with_behavior_policy(
                dataset_prompt=prompt,
                policy_rules=reduced_rules,
                placement=policy_placement,
            )
            variants.append((idx, rule, reduced_prompt))
        return full_prompt, variants

    units = split_prompt_into_loo_units(prompt)
    if len(units) < 2:
        return "", []

    seen_reduced_prompts: set[str] = set()
    variants = []
    full_prompt = (prompt or "").strip()
    for unit_idx, unit_text in enumerate(units):
        reduced_prompt = _remove_loo_unit(units, unit_idx)
        if not reduced_prompt or "Assistant:" not in reduced_prompt:
            continue
        if reduced_prompt in seen_reduced_prompts:
            continue
        seen_reduced_prompts.add(reduced_prompt)
        variants.append((unit_idx, unit_text, reduced_prompt))
    return full_prompt, variants


def run_leave_one_out_transfer_experiment(
    model: Any,
    dataset: Sequence[Example],
    scorer: PreferenceScorer,
    sampling_model: Optional[Any] = None,
    k_samples: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 256,
    top_p: float = 0.95,
    seed: int = 0,
    output_dir: str = f"experiments/preference_transfer",
    run_metadata: Optional[Dict[str, Any]] = None,
    dataset_manifest: Optional[List[Dict[str, Any]]] = None,
    intervention_mode: str = "prompt_units",
    policy_rules: Optional[Sequence[str]] = None,
    policy_placement: str = "prepend",
    reference_batch_size: int = 8,
    reference_mode: str = "dataset_margin",
    reference_samples: int = 8,
    reference_temperature: Optional[float] = None,
    reference_top_p: Optional[float] = None,
    reference_max_tokens: Optional[int] = None,
    reference_bucket_method: str = "top_bottom",
    reference_bucket_fraction: float = 0.5,
    reference_min_bucket_size: int = 1,
) -> Dict[str, Any]:
    """Run leave-one-out transfer correlation for a single model."""
    run_start = time.perf_counter()
    set_seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_unit_rows: List[Dict[str, Any]] = []
    skipped_examples: List[Dict[str, Any]] = []
    stage_timing: Dict[str, float] = {
        "reference_margin_s": 0.0,
        "reference_scorer_margin_s": 0.0,
        "sampling_s": 0.0,
        "generation_scoring_s": 0.0,
        "postprocess_s": 0.0,
        "io_write_s": 0.0,
        "total_s": 0.0,
    }
    stage_counts: Dict[str, float] = {
        "cache_hits": 0.0,
        "cache_misses": 0.0,
        "responses_total": 0.0,
        "responses_kept": 0.0,
    }

    mode = (intervention_mode or "prompt_units").strip().lower()
    if mode not in {"prompt_units", "policy_rules"}:
        raise ValueError("intervention_mode must be one of: prompt_units, policy_rules")

    reference_mode_key = (reference_mode or "dataset_margin").strip().lower()
    if reference_mode_key not in {"dataset_margin", "self_sampled"}:
        raise ValueError("reference_mode must be one of: dataset_margin, self_sampled")

    reference_temperature_value = float(reference_temperature) if reference_temperature is not None else float(temperature)
    reference_top_p_value = float(reference_top_p) if reference_top_p is not None else float(top_p)
    reference_max_tokens_value = int(reference_max_tokens) if reference_max_tokens is not None else int(max_tokens)

    resolved_policy_rules = [str(rule).strip() for rule in (policy_rules or DEFAULT_POLICY_RULES) if str(rule).strip()]
    if mode == "policy_rules" and len(resolved_policy_rules) < 2:
        raise ValueError("policy_rules intervention requires at least 2 non-empty rules")

    example_iterator = tqdm(enumerate(dataset), total=len(dataset), desc="LOO examples")
    generation_model = sampling_model if sampling_model is not None else model
    for example_idx, ex in example_iterator:
        full_prompt, loo_variants = build_leave_one_out_prompt_variants(
            prompt=ex.prompt,
            intervention_mode=mode,
            policy_rules=resolved_policy_rules,
            policy_placement=policy_placement,
        )
        if len(loo_variants) < 1:
            skipped_examples.append(
                {
                    "example_index": example_idx,
                    "reason": "insufficient_units",
                    "prompt_chars": len(ex.prompt),
                }
            )
            continue

        prompt_eval_cache: Dict[str, Dict[str, Any]] = {}
        shared_ref_responses: List[str] = []  # Shared pool for self_sampled mode

        prompts_for_ref = [full_prompt] + [variant[2] for variant in loo_variants]
        if reference_mode_key == "dataset_margin":
            t0 = time.perf_counter()
            ref_margins = compute_reference_margins_batched(
                model,
                prompts_for_ref,
                ex.chosen,
                ex.rejected,
                batch_size=reference_batch_size,
            )
            stage_timing["reference_margin_s"] += time.perf_counter() - t0
            for prompt_text, ref_margin in zip(prompts_for_ref, ref_margins):
                prompt_eval_cache[prompt_text] = {
                    "ref_margin": float(ref_margin),
                    "ref_mode": reference_mode_key,
                    "ref_pref_count": float("nan"),
                    "ref_unpref_count": float("nan"),
                    "ref_pref_responses": [],
                    "ref_unpref_responses": [],
                    "ref_pref_scores": [],
                    "ref_unpref_scores": [],
                }
        else:
            # For self_sampled mode with shared pools: sample once from full prompt
            t0 = time.perf_counter()
            if hasattr(generation_model, "generate_batch"):
                sampled_full = generation_model.generate_batch(
                    [full_prompt],
                    n_per_prompt=reference_samples,
                    max_new_tokens=reference_max_tokens_value,
                    temperature=reference_temperature_value,
                    top_p=reference_top_p_value,
                )
                shared_responses = sampled_full[0] if sampled_full else []
            else:
                shared_responses = sample_responses(
                    generation_model,
                    full_prompt,
                    k=reference_samples,
                    temperature=reference_temperature_value,
                    max_tokens=reference_max_tokens_value,
                    top_p=reference_top_p_value,
                )
            stage_timing["sampling_s"] += time.perf_counter() - t0
            stage_counts["responses_total"] += float(len(shared_responses))
            shared_ref_responses = list(shared_responses)  # Store for fallback path

            # Compute margins for full prompt and all reduced variants using shared pool
            for prompt_idx, prompt_text in enumerate(prompts_for_ref):
                if prompt_idx == 0:
                    # Full prompt: standalone margin
                    ref_result = _compute_self_sampled_reference_margin(
                        prompt=prompt_text,
                        responses=shared_responses,
                        scorer=scorer,
                        bucket_method=reference_bucket_method,
                        bucket_fraction=reference_bucket_fraction,
                        min_bucket_size=reference_min_bucket_size,
                    )
                else:
                    # Reduced prompt: score shared responses under reduced context
                    reduced_prompt = prompts_for_ref[prompt_idx]
                    marginal_pair = _compute_shared_pool_marginals(
                        full_prompt=full_prompt,
                        reduced_prompt=reduced_prompt,
                        shared_responses=shared_responses,
                        scorer=scorer,
                        bucket_method=reference_bucket_method,
                        bucket_fraction=reference_bucket_fraction,
                        min_bucket_size=reference_min_bucket_size,
                    )
                    # For reduced variant, use the "reduced" component of the pair
                    ref_result = marginal_pair["reduced"]

                stage_counts["responses_kept"] += float(len(ref_result["responses_used"]))
                prompt_eval_cache[prompt_text] = {
                    "ref_margin": float(ref_result["ref_margin"]),
                    "ref_mode": reference_mode_key,
                    "ref_pref_count": int(ref_result["pref_count"]),
                    "ref_unpref_count": int(ref_result["unpref_count"]),
                    "ref_pref_responses": list(ref_result["pref_responses"]),
                    "ref_unpref_responses": list(ref_result["unpref_responses"]),
                    "ref_pref_scores": list(ref_result["pref_scores"]),
                    "ref_unpref_scores": list(ref_result["unpref_scores"]),
                    "ref_responses_all": list(ref_result["responses_all"]),
                    "ref_responses_used": list(ref_result["responses_used"]),
                }

        # Pre-batch all generation for this example when the backend supports it
        # (e.g. vLLM with n>1 per prompt is far more efficient than k serial calls)
        if hasattr(generation_model, "generate_batch"):
            t0 = time.perf_counter()
            all_gen_responses = generation_model.generate_batch(
                prompts_for_ref,
                n_per_prompt=k_samples,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            stage_timing["sampling_s"] += time.perf_counter() - t0
            for gp, gr in zip(prompts_for_ref, all_gen_responses):
                prompt_eval_cache[gp]["_prefetched_responses"] = gr

        def evaluate_prompt(prompt_text: str) -> Dict[str, Any]:
            cached = prompt_eval_cache.get(prompt_text)
            if cached is not None and "gen_score" in cached:
                stage_counts["cache_hits"] += 1.0
                return cached
            if cached is None:
                stage_counts["cache_misses"] += 1.0
                cached = {}
                if reference_mode_key == "dataset_margin":
                    t0 = time.perf_counter()
                    cached["ref_margin"] = float(compute_reference_margin(model, prompt_text, ex.chosen, ex.rejected))
                    stage_timing["reference_margin_s"] += time.perf_counter() - t0
                    cached["ref_mode"] = reference_mode_key
                    cached["ref_pref_count"] = float("nan")
                    cached["ref_unpref_count"] = float("nan")
                    cached["ref_pref_responses"] = []
                    cached["ref_unpref_responses"] = []
                    cached["ref_pref_scores"] = []
                    cached["ref_unpref_scores"] = []
                else:
                    # Use shared pool if available, otherwise fall back to fresh sampling
                    t0 = time.perf_counter()
                    if shared_ref_responses:
                        # Use pre-sampled shared pool, compute marginals for this specific prompt
                        if prompt_text == full_prompt:
                            ref_result = _compute_self_sampled_reference_margin(
                                prompt=prompt_text,
                                responses=shared_ref_responses,
                                scorer=scorer,
                                bucket_method=reference_bucket_method,
                                bucket_fraction=reference_bucket_fraction,
                                min_bucket_size=reference_min_bucket_size,
                            )
                        else:
                            # Reduced prompt: use shared pool marginals
                            marginal_pair = _compute_shared_pool_marginals(
                                full_prompt=full_prompt,
                                reduced_prompt=prompt_text,
                                shared_responses=shared_ref_responses,
                                scorer=scorer,
                                bucket_method=reference_bucket_method,
                                bucket_fraction=reference_bucket_fraction,
                                min_bucket_size=reference_min_bucket_size,
                            )
                            ref_result = marginal_pair["reduced"]
                    else:
                        # Fallback: fresh sampling (only if shared pool wasn't created)
                        ref_responses = sample_responses(
                            generation_model,
                            prompt_text,
                            k=reference_samples,
                            temperature=reference_temperature_value,
                            max_tokens=reference_max_tokens_value,
                            top_p=reference_top_p_value,
                        )
                        stage_counts["responses_total"] += float(len(ref_responses))
                        ref_result = _compute_self_sampled_reference_margin(
                            prompt=prompt_text,
                            responses=ref_responses,
                            scorer=scorer,
                            bucket_method=reference_bucket_method,
                            bucket_fraction=reference_bucket_fraction,
                            min_bucket_size=reference_min_bucket_size,
                        )
                    stage_timing["reference_scorer_margin_s"] += time.perf_counter() - t0
                    stage_counts["responses_kept"] += float(len(ref_result["responses_used"]))
                    cached["ref_margin"] = float(ref_result["ref_margin"])
                    cached["ref_mode"] = reference_mode_key
                    cached["ref_pref_count"] = int(ref_result["pref_count"])
                    cached["ref_unpref_count"] = int(ref_result["unpref_count"])
                    cached["ref_pref_responses"] = list(ref_result["pref_responses"])
                    cached["ref_unpref_responses"] = list(ref_result["unpref_responses"])
                    cached["ref_pref_scores"] = list(ref_result["pref_scores"])
                    cached["ref_unpref_scores"] = list(ref_result["unpref_scores"])
                    cached["ref_responses_all"] = list(ref_result["responses_all"])
                    cached["ref_responses_used"] = list(ref_result["responses_used"])
            else:
                stage_counts["cache_hits"] += 1.0

            prefetched = cached.get("_prefetched_responses")
            if prefetched is not None:
                responses = prefetched
                stage_counts["responses_total"] += float(len(responses))
            else:
                t0 = time.perf_counter()
                responses = sample_responses(
                    generation_model,
                    prompt_text,
                    k=k_samples,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                stage_timing["sampling_s"] += time.perf_counter() - t0
                stage_counts["responses_total"] += float(len(responses))
            kept_responses: List[str] = list(responses)
            stage_counts["responses_kept"] += float(len(kept_responses))
            t0 = time.perf_counter()
            scores: List[float] = scorer.score_batch(prompt_text, kept_responses) if kept_responses else []
            stage_timing["generation_scoring_s"] += time.perf_counter() - t0
            gen_score = float(np.mean(scores)) if scores else float("nan")

            t0 = time.perf_counter()
            scorer_ref_margin = compute_scorer_reference_margin(scorer, prompt_text, ex.chosen, ex.rejected)
            stage_timing["reference_scorer_margin_s"] += time.perf_counter() - t0

            result = {
                "ref_margin": float(cached["ref_margin"]),
                "ref_mode": cached.get("ref_mode", reference_mode_key),
                "ref_pref_count": cached.get("ref_pref_count", float("nan")),
                "ref_unpref_count": cached.get("ref_unpref_count", float("nan")),
                "ref_pref_responses": list(cached.get("ref_pref_responses", [])),
                "ref_unpref_responses": list(cached.get("ref_unpref_responses", [])),
                "ref_pref_scores": list(cached.get("ref_pref_scores", [])),
                "ref_unpref_scores": list(cached.get("ref_unpref_scores", [])),
                "ref_responses_all": list(cached.get("ref_responses_all", [])),
                "ref_responses_used": list(cached.get("ref_responses_used", [])),
                "scorer_ref_margin": float(scorer_ref_margin),
                "gen_score": float(gen_score),
                "responses_all": responses,
                "responses_used": kept_responses,
            }
            prompt_eval_cache[prompt_text] = result
            return result

        full_eval = evaluate_prompt(full_prompt)
        full_ref_margin = float(full_eval["ref_margin"])
        full_ref_scorer_margin = float(full_eval["scorer_ref_margin"])
        full_gen_score = float(full_eval["gen_score"])
        full_responses = list(full_eval["responses_all"])
        full_kept_responses = list(full_eval["responses_used"])
        full_ref_responses = list(full_eval.get("ref_responses_all", []))
        full_ref_kept_responses = list(full_eval.get("ref_responses_used", []))
        full_ref_pref_responses = list(full_eval.get("ref_pref_responses", []))
        full_ref_unpref_responses = list(full_eval.get("ref_unpref_responses", []))
        full_ref_pref_scores = list(full_eval.get("ref_pref_scores", []))
        full_ref_unpref_scores = list(full_eval.get("ref_unpref_scores", []))

        for unit_idx, unit_text, reduced_prompt in loo_variants:
            reduced_eval = evaluate_prompt(reduced_prompt)
            reduced_ref_margin = float(reduced_eval["ref_margin"])
            reduced_ref_scorer_margin = float(reduced_eval["scorer_ref_margin"])
            delta_ref = reduced_ref_margin - full_ref_margin
            delta_ref_scorer = reduced_ref_scorer_margin - full_ref_scorer_margin

            reduced_gen_score = float(reduced_eval["gen_score"])
            reduced_responses = list(reduced_eval["responses_all"])
            reduced_kept_responses = list(reduced_eval["responses_used"])
            delta_gen = reduced_gen_score - full_gen_score
            reduced_ref_responses = list(reduced_eval.get("ref_responses_all", []))
            reduced_ref_kept_responses = list(reduced_eval.get("ref_responses_used", []))
            reduced_ref_pref_responses = list(reduced_eval.get("ref_pref_responses", []))
            reduced_ref_unpref_responses = list(reduced_eval.get("ref_unpref_responses", []))
            reduced_ref_pref_scores = list(reduced_eval.get("ref_pref_scores", []))
            reduced_ref_unpref_scores = list(reduced_eval.get("ref_unpref_scores", []))

            per_unit_rows.append(
                {
                    "example_index": example_idx,
                    "unit_index": unit_idx,
                    "num_units": len(loo_variants),
                    "delta_ref": float(delta_ref),
                    "delta_ref_scorer": float(delta_ref_scorer),
                    "delta_gen": float(delta_gen),
                    "full_ref_margin": float(full_ref_margin),
                    "reduced_ref_margin": float(reduced_ref_margin),
                    "full_ref_scorer_margin": float(full_ref_scorer_margin),
                    "reduced_ref_scorer_margin": float(reduced_ref_scorer_margin),
                    "full_gen_score": float(full_gen_score),
                    "reduced_gen_score": float(reduced_gen_score),
                    "full_generations_all": full_responses,
                    "full_generations_used": full_kept_responses,
                    "reduced_generations_all": reduced_responses,
                    "reduced_generations_used": reduced_kept_responses,
                    "full_reference_responses_all": full_ref_responses,
                    "full_reference_responses_used": full_ref_kept_responses,
                    "full_reference_preferred_responses": full_ref_pref_responses,
                    "full_reference_unpreferred_responses": full_ref_unpref_responses,
                    "full_reference_preferred_scores": full_ref_pref_scores,
                    "full_reference_unpreferred_scores": full_ref_unpref_scores,
                    "reduced_reference_responses_all": reduced_ref_responses,
                    "reduced_reference_responses_used": reduced_ref_kept_responses,
                    "reduced_reference_preferred_responses": reduced_ref_pref_responses,
                    "reduced_reference_unpreferred_responses": reduced_ref_unpref_responses,
                    "reduced_reference_preferred_scores": reduced_ref_pref_scores,
                    "reduced_reference_unpreferred_scores": reduced_ref_unpref_scores,
                    "dataset_prompt": ex.prompt,
                    "full_prompt": full_prompt,
                    "reduced_prompt": reduced_prompt,
                    "reference_mode": reference_mode_key,
                    "reference_bucket_method": reference_bucket_method,
                    "reference_bucket_fraction": float(reference_bucket_fraction),
                    "full_reference_pref_count": full_eval.get("ref_pref_count", float("nan")),
                    "full_reference_unpref_count": full_eval.get("ref_unpref_count", float("nan")),
                    "reduced_reference_pref_count": reduced_eval.get("ref_pref_count", float("nan")),
                    "reduced_reference_unpref_count": reduced_eval.get("ref_unpref_count", float("nan")),
                    "unit_text": unit_text,
                    "unit_chars": len(unit_text),
                    "unit_kind": "policy_rule" if mode == "policy_rules" else "prompt_unit",
                }
            )

    postprocess_start = time.perf_counter()
    delta_ref_list = [float(row["delta_ref"]) for row in per_unit_rows]
    delta_ref_scorer_list = [float(row["delta_ref_scorer"]) for row in per_unit_rows]
    delta_gen_list = [float(row["delta_gen"]) for row in per_unit_rows]
    raw_corr = compute_transfer_correlation(delta_ref_list, delta_gen_list)
    raw_corr_ref_scorer = compute_transfer_correlation(delta_ref_scorer_list, delta_gen_list)
    delta_ref_z_list, delta_gen_z_list, z_meta = standardize_deltas_per_example(per_unit_rows)
    example_indices = [int(row.get("example_index", -1)) for row in per_unit_rows]
    delta_ref_scorer_z_list, delta_gen_for_ref_scorer_z_list, z_meta_ref_scorer = standardize_pair_lists_per_example(
        example_indices,
        delta_ref_scorer_list,
        delta_gen_list,
    )
    for row_idx, row in enumerate(per_unit_rows):
        row["delta_ref_z"] = float(delta_ref_z_list[row_idx])
        row["delta_gen_z"] = float(delta_gen_z_list[row_idx])
        row["delta_ref_scorer_z"] = float(delta_ref_scorer_z_list[row_idx])
        row["delta_gen_z_for_ref_scorer"] = float(delta_gen_for_ref_scorer_z_list[row_idx])
    corr = compute_transfer_correlation(delta_ref_z_list, delta_gen_z_list)
    corr_ref_scorer = compute_transfer_correlation(delta_ref_scorer_z_list, delta_gen_for_ref_scorer_z_list)

    scatter_path = out_dir / "scatter_leave_one_out.png"
    save_transfer_scatter(
        delta_ref_z_list,
        delta_gen_z_list,
        output_path=scatter_path,
        title="Leave-One-Out: per-example z(Δref) vs z(Δgen)",
    )
    stage_timing["postprocess_s"] += time.perf_counter() - postprocess_start

    mean_delta_ref = float(np.nanmean(delta_ref_list)) if delta_ref_list else float("nan")
    valid_delta_gen = [x for x in delta_gen_list if not math.isnan(float(x))]
    mean_delta_gen = float(np.nanmean(valid_delta_gen)) if valid_delta_gen else float("nan")

    summary_row = {
        "intervention": "leave_one_out_policy_rules" if mode == "policy_rules" else "leave_one_out",
        "n_examples": float(len(dataset)),
        "n_examples_skipped": float(len(skipped_examples)),
        "k_samples": float(k_samples),
        "mean_delta_ref": mean_delta_ref,
        "mean_delta_gen": mean_delta_gen,
        "pearson_r": corr["pearson_r"],
        "pearson_p": corr["pearson_p"],
        "spearman_rho": corr["spearman_rho"],
        "spearman_p": corr["spearman_p"],
        "n": corr["n"],
        "correlation_mode": "per_example_zscore",
        "n_examples_standardized": z_meta["n_examples_standardized"],
        "pearson_r_ref_scorer": corr_ref_scorer["pearson_r"],
        "pearson_p_ref_scorer": corr_ref_scorer["pearson_p"],
        "spearman_rho_ref_scorer": corr_ref_scorer["spearman_rho"],
        "spearman_p_ref_scorer": corr_ref_scorer["spearman_p"],
        "n_ref_scorer": corr_ref_scorer["n"],
        "n_examples_standardized_ref_scorer": z_meta_ref_scorer["n_examples_standardized"],
        "pearson_r_raw": raw_corr["pearson_r"],
        "pearson_p_raw": raw_corr["pearson_p"],
        "spearman_rho_raw": raw_corr["spearman_rho"],
        "spearman_p_raw": raw_corr["spearman_p"],
        "n_raw": raw_corr["n"],
        "pearson_r_ref_scorer_raw": raw_corr_ref_scorer["pearson_r"],
        "pearson_p_ref_scorer_raw": raw_corr_ref_scorer["pearson_p"],
        "spearman_rho_ref_scorer_raw": raw_corr_ref_scorer["spearman_rho"],
        "spearman_p_ref_scorer_raw": raw_corr_ref_scorer["spearman_p"],
        "n_ref_scorer_raw": raw_corr_ref_scorer["n"],
    }

    io_start = time.perf_counter()
    csv_path = out_dir / "results.csv"
    _write_csv_fallback(csv_path, [summary_row])

    config_payload = {
        "mode": "leave_one_out",
        "intervention_mode": mode,
        "policy_placement": policy_placement,
        "policy_rules": resolved_policy_rules,
        "reference_mode": reference_mode_key,
        "reference_samples": int(reference_samples),
        "reference_temperature": float(reference_temperature_value),
        "reference_top_p": float(reference_top_p_value),
        "reference_max_tokens": int(reference_max_tokens_value),
        "reference_bucket_method": reference_bucket_method,
        "reference_bucket_fraction": float(reference_bucket_fraction),
        "reference_min_bucket_size": int(reference_min_bucket_size),
        "k_samples": k_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "seed": seed,
    }
    config_path = out_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    detailed_json_path = out_dir / "results_detailed.json"
    detailed_payload = {
        "config": config_payload,
        "summary_rows": [summary_row],
        "run_metadata": run_metadata or {},
        "dataset_manifest": dataset_manifest or [],
        "dataset_size": len(dataset),
        "per_unit_rows": per_unit_rows,
        "skipped_examples": skipped_examples,
        "timing": {
            "stage_seconds": stage_timing,
            "counts": stage_counts,
        },
        "scatter_path": str(scatter_path),
    }
    with detailed_json_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_payload, f, indent=2)
    stage_timing["io_write_s"] += time.perf_counter() - io_start
    stage_timing["total_s"] = time.perf_counter() - run_start
    summary_row["timing_total_s"] = stage_timing["total_s"]
    summary_row["timing_sampling_s"] = stage_timing["sampling_s"]
    summary_row["timing_reference_margin_s"] = stage_timing["reference_margin_s"]
    summary_row["timing_reference_scorer_margin_s"] = stage_timing["reference_scorer_margin_s"]
    summary_row["timing_generation_scoring_s"] = stage_timing["generation_scoring_s"]
    summary_row["timing_postprocess_s"] = stage_timing["postprocess_s"]
    summary_row["timing_io_write_s"] = stage_timing["io_write_s"]

    _write_csv_fallback(csv_path, [summary_row])
    detailed_payload["summary_rows"] = [summary_row]
    detailed_payload["timing"] = {
        "stage_seconds": stage_timing,
        "counts": stage_counts,
    }
    with detailed_json_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_payload, f, indent=2)

    print(
        "[timing] total={total:.2f}s ref={ref:.2f}s sample={sample:.2f}s "
        "score={score:.2f}s post={post:.2f}s io={io:.2f}s cache={hits:.0f}/{misses:.0f}".format(
            total=stage_timing["total_s"],
            ref=stage_timing["reference_margin_s"],
            sample=stage_timing["sampling_s"],
            score=stage_timing["generation_scoring_s"],
            post=stage_timing["postprocess_s"],
            io=stage_timing["io_write_s"],
            hits=stage_counts["cache_hits"],
            misses=stage_counts["cache_misses"],
        )
    )

    return {
        "rows": [summary_row],
        "results_csv": str(csv_path),
        "config_path": str(config_path),
        "detailed_json_path": str(detailed_json_path),
        "scatter_path": str(scatter_path),
        "correlation": corr,
    }


def _write_csv_fallback(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import csv

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
