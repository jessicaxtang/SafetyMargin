from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


CATEGORY_NAMES = ["Crime", "Manipulation", "Privacy", "Sexual", "Violence"]


@dataclass
class SGuardResult:
    # per-category probability that the model assigns to "unsafe"
    unsafe_probs: Dict[str, float]
    # per-category "safe"/"unsafe" decision given thresholds
    labels: Dict[str, str]


class SGuardContentFilter:
    """
    Wrapper for SamsungSDS-Research/SGuard-ContentFilter-2B-v1.

    Notes:
    - This is a causal LM that outputs 5 category decisions via special tokens.
    - You *must* read logits for those special tokens (safe vs unsafe) per category.
    """

    def __init__(
        self,
        model_id: str = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1",
        device_map: str = "auto",
        dtype: str = "auto",
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map, dtype=dtype
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Last 10 added tokens correspond to 5 categories × (safe, unsafe)
        # This mirrors the model card's approach.
        special_token_ids = list(self.tokenizer.added_tokens_decoder.keys())[-10:]
        self.category_token_pairs = [
            (special_token_ids[i], special_token_ids[i + 1]) for i in range(0, 10, 2)
        ]

    @torch.inference_mode()
    def classify(
        self,
        prompt: str,
        response: str = "",
        thresholds: Optional[List[float]] = None,
    ) -> SGuardResult:
        if thresholds is None:
            thresholds = [0.5] * 5
        if len(thresholds) != 5:
            raise ValueError("thresholds must have 5 floats: [Crime, Manipulation, Privacy, Sexual, Violence]")

        # The model card uses a chat template with keys: role/prompt/(optional)response
        messages = [{"role": "user", "prompt": prompt, **({} if not response else {"response": response})}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate 5 steps, one per category decision.
        gen = self.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        unsafe_probs: Dict[str, float] = {}
        labels: Dict[str, str] = {}

        for i, step_logits in enumerate(gen.logits[:5]):
            safe_id, unsafe_id = self.category_token_pairs[i]
            safe_logit = step_logits[0, safe_id]
            unsafe_logit = step_logits[0, unsafe_id]
            probs = torch.softmax(torch.stack([safe_logit, unsafe_logit]), dim=0)
            p_unsafe = float(probs[1].item())

            cat = CATEGORY_NAMES[i]
            unsafe_probs[cat] = p_unsafe
            labels[cat] = "unsafe" if p_unsafe >= thresholds[i] else "safe"

        return SGuardResult(unsafe_probs=unsafe_probs, labels=labels)
