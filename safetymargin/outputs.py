"""Model generation and output extraction utilities."""

from typing import Any, Dict, List, Optional

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def extract_generated_text(output: Any, model: Any, prompt_text: str) -> Optional[str]:
    """Extract generated text from various model output formats.
    
    Args:
        output: Model output (dict, list, tensor, or object with generated_text)
        model: Model instance (for tokenizer access)
        prompt_text: Original prompt to strip from output
        
    Returns:
        Extracted text with prompt stripped, or None if extraction fails
    """
    tokenizer = getattr(model, "tokenizer", None)

    def strip_prompt(text: str) -> str:
        text = text.strip()
        prompt_clean = prompt_text.strip()
        if prompt_clean and text.startswith(prompt_clean):
            remainder = text[len(prompt_clean):].lstrip()
            if remainder:
                return remainder
        return text

    if isinstance(output, dict):
        text = output.get("generated_text") or output.get("summary_text")
        if text:
            return strip_prompt(str(text)) or None
        seq = output.get("sequences") or output.get("generated_ids")
        if seq is not None and tokenizer is not None:
            try:
                ids = seq[0] if isinstance(seq, (list, tuple)) else seq
                decoded = tokenizer.decode(ids, skip_special_tokens=True)
                return strip_prompt(decoded) or None
            except Exception:
                pass

    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            text = first.get("generated_text") or first.get("summary_text")
            if text:
                return strip_prompt(str(text)) or None
            seq = first.get("sequences") or first.get("generated_ids")
            if seq is not None and tokenizer is not None:
                try:
                    ids = seq[0] if isinstance(seq, (list, tuple)) else seq
                    decoded = tokenizer.decode(ids, skip_special_tokens=True)
                    return strip_prompt(decoded) or None
                except Exception:
                    pass

    text_attr = getattr(output, "generated_text", None)
    if text_attr is not None:
        if isinstance(text_attr, list):
            if not text_attr:
                return None
            text_str = text_attr[0]
        else:
            text_str = text_attr
        return strip_prompt(str(text_str)) or None

    if tokenizer is None:
        return None

    sequences = None
    if hasattr(output, "sequences"):
        sequences = output.sequences
    elif hasattr(output, "generated_ids"):
        sequences = output.generated_ids
    elif HAS_TORCH and isinstance(output, torch.Tensor):
        sequences = output

    if sequences is None:
        return None

    try:
        if HAS_TORCH and hasattr(sequences, "dim"):
            seq_tensor = sequences
            if seq_tensor.dim() == 2:
                seq_tensor = seq_tensor[0]
            decoded = tokenizer.decode(seq_tensor, skip_special_tokens=True)
        else:
            seq = sequences[0] if isinstance(sequences, list) and sequences else sequences
            if seq is None:
                return None
            decoded = tokenizer.decode(seq, skip_special_tokens=True)
        return strip_prompt(decoded) or None
    except Exception:
        return None


def _split_ctx_to_chat(ctx: str) -> List[Dict[str, str]]:
    """Convert context string to chat messages format.
    
    Lines starting with SYSTEM/POLICY/SAFETY become system messages.
    All other lines become user messages.
    """
    sys_lines: List[str] = []
    user_lines: List[str] = []
    for raw in (ctx or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        head = line.split(":", 1)[0].upper()
        if head in {"SYSTEM", "SYSTEM POLICY", "POLICY", "SAFETY"}:
            sys_lines.append(line)
        else:
            user_lines.append(line)
    messages: List[Dict[str, str]] = []
    if sys_lines:
        messages.append({"role": "system", "content": "\n".join(sys_lines)})
    user_text = "\n".join(user_lines) if user_lines else ctx
    messages.append({"role": "user", "content": user_text})
    return messages


def generate_outputs(
    model: Any,
    prompt: str,
    max_new_tokens: int,
    num_stochastic: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Optional[Dict[str, Any]]:
    """Generate both greedy and stochastic outputs from a model.
    
    Args:
        model: Model with generate() method
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        num_stochastic: Number of stochastic samples to generate
        temperature: Sampling temperature for stochastic generation
        top_p: Nucleus sampling parameter
        
    Returns:
        Dictionary with 'greedy' and 'stochastic' keys containing outputs,
        or None if model doesn't support generation
    """
    if not hasattr(model, "generate"):
        return None
    outputs: Dict[str, Any] = {"greedy": None, "stochastic": []}

    # If the model exposes a chat-aware tokenizer, render the prompt via the
    # official chat template to avoid formatting bugs that cause empty outputs.
    rendered_prompt = prompt
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            # Preserve assistant header by keeping the tail of long prompts
            try:
                setattr(tokenizer, "truncation_side", "left")
                setattr(tokenizer, "padding_side", "left")
            except Exception:
                pass
            chat = _split_ctx_to_chat(prompt)
            rendered_prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered_prompt = prompt

    # Minimal prompt preview for debugging
    try:
        preview = rendered_prompt[:60].replace("\n", " ") if isinstance(rendered_prompt, str) else ""
        print(f"[preview] prompt[:60]: {preview}")
    except Exception:
        pass

    # Generate greedy output
    try:
        greedy_out = model.generate(
            rendered_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            return_log_probs=False,
        )
        greedy_text = getattr(greedy_out, "generated_text", None)
        if not greedy_text:
            greedy_text = extract_generated_text(greedy_out, model, prompt)
        if greedy_text:
            greedy_text = greedy_text.strip()
        outputs["greedy"] = greedy_text or None
    except Exception:
        outputs["greedy"] = None

    # Generate stochastic samples
    for _ in range(max(num_stochastic, 0)):
        try:
            sample_out = model.generate(
                rendered_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                return_log_probs=False,
            )
            sample_text = getattr(sample_out, "generated_text", None)
            if not sample_text:
                sample_text = extract_generated_text(sample_out, model, prompt)
            if sample_text:
                sample_text = sample_text.strip()
            outputs.setdefault("stochastic", []).append(sample_text or None)
        except Exception:
            outputs.setdefault("stochastic", []).append(None)
    return outputs
