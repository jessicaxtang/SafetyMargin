"""vLLM model wrapper implementation focused on fast text generation."""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from safetymargin.models.base import ModelWrapper, ModelOutput


class VLLMModelWrapper(ModelWrapper):
    """Wrapper around vLLM for sampling-heavy workloads.

    This wrapper intentionally focuses on generation. Teacher-forced log-prob
    APIs used by attribution remain unsupported here.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        use_cache: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        torch_dtype: str = "auto",
        auto_load: bool = True,
    ):
        super().__init__(model_name=model_name, device=device, max_length=max_length, use_cache=use_cache)
        self.tensor_parallel_size = max(int(tensor_parallel_size), 1)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.torch_dtype = torch_dtype
        self.llm = None
        if auto_load:
            self.load()

    def load(self):
        try:
            from vllm import LLM
        except Exception as exc:
            raise RuntimeError("vLLM is not available. Install/verify the `vllm` package.") from exc

        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype=self.torch_dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_length,
            tokenizer=self.model_name,
        )
        self.model = self.llm

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_log_probs: bool = True,
    ) -> ModelOutput:
        if self.llm is None:
            self.load()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=max(float(temperature), 0.0),
            top_p=float(top_p),
            logprobs=1 if return_log_probs else None,
        )

        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        if not outputs:
            return ModelOutput(generated_text="", metadata={"num_generated_tokens": 0, "prompt_length": 0})

        first = outputs[0]
        completion = first.outputs[0] if first.outputs else None
        generated_text = completion.text if completion is not None else ""

        token_log_probs: Optional[torch.Tensor] = None
        if return_log_probs and completion is not None and getattr(completion, "logprobs", None):
            gathered = []
            for token_id, token_candidates in zip(completion.token_ids, completion.logprobs):
                if token_candidates is None:
                    continue
                token_entry = token_candidates.get(token_id)
                if token_entry is None:
                    continue
                gathered.append(float(token_entry.logprob))
            if gathered:
                token_log_probs = torch.tensor(gathered, dtype=torch.float32).unsqueeze(0)

        return ModelOutput(
            generated_text=generated_text,
            token_log_probs=token_log_probs,
            metadata={
                "num_generated_tokens": len(getattr(completion, "token_ids", []) or []),
                "prompt_length": len(getattr(first, "prompt_token_ids", []) or []),
            },
        )

    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[List[str]]:
        """Generate n_per_prompt responses for each prompt in a single batched vLLM call.

        Uses vLLM's native n>1 parameter so each prompt is processed once regardless
        of n_per_prompt, providing significant throughput gains over serial generate().

        Returns:
            List of length len(prompts), each element is a list of n_per_prompt strings.
        """
        if self.llm is None:
            self.load()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=max(float(temperature), 1e-6 if temperature == 0.0 else float(temperature)),
            top_p=float(top_p),
            n=max(int(n_per_prompt), 1),
        )

        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        results: List[List[str]] = []
        for out in outputs:
            texts = [(o.text or "").strip() if o is not None else "" for o in out.outputs]
            results.append(texts)
        return results

    def get_log_probs(
        self,
        prompt: str,
        continuation: Optional[str] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("VLLMModelWrapper does not support teacher-forced get_log_probs().")

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.tolist()
        else:
            ids = token_ids
        return self.tokenizer.decode(ids, skip_special_tokens=True)
