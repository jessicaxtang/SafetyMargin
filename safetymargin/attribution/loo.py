"""
Leave-One-Out (LOO) attribution method.
"""

import numpy as np
from typing import Optional
from tqdm import tqdm

from safetymargin.attribution.base import AttributionMethod, AttributionResult
from safetymargin.datasets.prompt_example import PromptExample, ContextSpan
from safetymargin.models.base import ModelWrapper


class LOOMethod(AttributionMethod):
    """
    Leave-One-Out attribution method.
    
    This method measures each span's importance by comparing the model's output
    with and without that span. The attribution score is based on how much
    the output changes (in terms of log probability) when a span is removed.
    
    Algorithm:
    1. Get baseline output with full prompt: P(output | full_prompt)
    2. For each span i:
        a. Remove span i from prompt
        b. Get output: P(output | prompt without span_i)
        c. Score_i = P(full) - P(without_i)
    
    Higher scores indicate more influential spans.
    
    Attributes:
        model: Model wrapper for inference
        max_new_tokens: Maximum tokens to generate
        use_continuation: Whether to score a fixed continuation
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        max_new_tokens: int = 50,
        use_continuation: bool = False,
        use_teacher_forcing: bool = False,
        preserve_prompt_structure: bool = False,
    ):
        """
        Initialize LOO method.
        
        Args:
            model: Model wrapper
            max_new_tokens: Maximum tokens to generate for output
            use_continuation: If True, score a fixed continuation instead of generating
            use_teacher_forcing: If True, score log probs of a fixed continuation (requires example.ground_truth)
            preserve_prompt_structure: If True, mask spans instead of removing them to keep formatting identical
        """
        super().__init__(model, name="LOO")
        self.max_new_tokens = max_new_tokens
        self.use_continuation = use_continuation
        self.use_teacher_forcing = use_teacher_forcing
        self.preserve_prompt_structure = preserve_prompt_structure
    
    def run(self, example: PromptExample) -> AttributionResult:
        """
        Run Leave-One-Out attribution analysis.
        
        Args:
            example: The prompt example to analyze
            
        Returns:
            AttributionResult with span attribution scores
        """
        target_text = None
        if self.use_teacher_forcing:
            target_text = (example.ground_truth or "").rstrip("\n") + "\n"
            if not target_text.strip():
                raise ValueError(
                    "Teacher forcing requested for LOO but example.ground_truth is empty."
                )
        
        print("[LOO] Full prompt:")
        print(example.full_prompt)
        
        if target_text is not None:
            baseline_sum_log_prob, token_log_probs = self._score_with_teacher_forcing(
                example.full_prompt,
                target_text,
            )
            generated_text = target_text
        else:
            # Get baseline output with full prompt
            baseline_output = self.model.generate(
                example.full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,  # Greedy for consistency
                return_log_probs=True,
            )
            generated_text = baseline_output.generated_text
            baseline_sum_log_prob = self._compute_score(baseline_output)
            token_log_probs = (
                baseline_output.token_log_probs.cpu().numpy()
                if baseline_output.token_log_probs is not None
                else None
            )
        print(f"[LOO] Baseline log prob: {baseline_sum_log_prob:.4f}")

        # Compute attribution for each span
        span_scores = []

        for span_idx, span in enumerate(tqdm(example.spans, desc="LOO Attribution", unit="span")):
            print(f"\n[LOO] Span {span_idx}: '{span.text.strip()}'")
            if self.preserve_prompt_structure:
                ablated_prompt = self._get_masked_prompt(example.full_prompt, span)
            else:
                ablated_prompt = example.get_prompt_without_span(span_idx)

            if target_text is not None:
                ablated_log_prob, _ = self._score_with_teacher_forcing(
                    ablated_prompt,
                    target_text,
                )
            elif self.use_continuation and generated_text:
                ablated_log_prob = self._score_continuation(ablated_prompt, generated_text)
            else:
                ablated_output = self.model.generate(
                    ablated_prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    return_log_probs=True,
                )
                ablated_log_prob = self._compute_score(ablated_output)
            # print(f"[LOO] Ablated log prob: {ablated_log_prob:.4f}")

            attribution = baseline_sum_log_prob - ablated_log_prob
            # print(f"[LOO] Attribution score: {attribution:.4f}")
            span_scores.append(attribution)

        span_scores = np.array(span_scores)

        source_scores = self.aggregate_by_source(example, span_scores)

        print("[LOO] Final span scores:", span_scores)

        return AttributionResult(
            example_id=example.id,
            span_scores=span_scores,
            source_scores=source_scores,
            predicted_output=generated_text,
            token_scores=token_log_probs,
            method_name=self.name,
            metadata={
                "baseline_sum_log_prob": baseline_sum_log_prob,
                "num_ablations": len(span_scores),
                "use_teacher_forcing": self.use_teacher_forcing,
                "preserve_prompt_structure": self.preserve_prompt_structure,
            }
        )
    
    def _compute_score(self, output) -> float:
        """
        Compute a scalar score from model output.
        
        Uses the mean log probability of generated tokens.
        
        Args:
            output: ModelOutput from generation
            
        Returns:
            Scalar score (mean log prob)
        """
        if output.token_log_probs is not None:
            return float(output.token_log_probs.mean().item())
        return 0.0
    
    def _score_continuation(self, prompt: str, continuation: str) -> float:
        """
        Score a specific continuation given a prompt.
        
        Args:
            prompt: Input prompt
            continuation: Text to score
            
        Returns:
            Mean log probability of continuation
        """
        log_probs = self.model.get_log_probs(prompt, continuation)
        return float(log_probs.mean().item())

    def _score_with_teacher_forcing(
        self,
        prompt: str,
        target_text: str,
    ) -> tuple[float, np.ndarray]:
        """
        Compute mean log probability for a fixed continuation using teacher forcing.
        
        Args:
            prompt: Prompt text
            target_text: Continuation to score
        
        Returns:
            Tuple of (mean log prob, numpy array of per-token log probs)
        """
        result = self.model.teacher_forcing_forward(prompt, target_text)
        log_probs = result["target_log_probs"]
        if log_probs is None or log_probs.nelement() == 0:
            return 0.0, np.array([])
        log_probs_np = log_probs.detach().cpu().numpy()
        return result["sum_log_prob"], log_probs_np

    def _get_masked_prompt(self, full_prompt: str, span: ContextSpan) -> str:
        """
        Replace characters covered by the span with whitespace/newlines to preserve layout.
        
        Args:
            full_prompt: Original prompt string
            span: Span to mask
        
        Returns:
            Prompt with the span masked out
        """
        span_text = span.text
        trailing_newlines = ""
        stripped = span_text.rstrip("\n")
        if len(stripped) != len(span_text):
            trailing_newlines = span_text[len(stripped):]
        replacement = "[REMOVED]" + trailing_newlines
        return full_prompt[: span.start_idx] + replacement + full_prompt[span.end_idx:]


class LOOAttributor(LOOMethod):
    """Compatibility alias for AOI API parity."""
    pass
