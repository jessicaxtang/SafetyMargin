#!/usr/bin/env python3
"""
Quick generation sanity check for chat-tuned models.

Usage examples:
  python scripts/test_generation.py \
    --model meta-llama/Llama-3.2-1B-Instruct --device cpu

Optional env:
  HUGGINGFACE_TOKEN / HUGGINGFACE_HUB_TOKEN for gated models.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal chat generation test")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt", type=str, help="Override user content (default: Say 'ok'.)")
    p.add_argument("--system", type=str, default="You are helpful.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from safetymargin.models.huggingface_wrapper import HFModel
    except Exception as exc:  # transformers/torch missing
        print(f"[test_generation] Could not import HFModel: {exc}")
        return 2

    # Optional login for private models
    try:
        tok = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if tok:
            from huggingface_hub import login
            login(token=tok)
    except Exception:
        pass

    model = HFModel(model_name=args.model, auto_load=True, device=args.device)
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        print("[test_generation] Tokenizer not available after load.")
        return 3

    # Build a tiny chat
    user_content = args.prompt or "Say 'ok'."
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": user_content},
    ]

    # Render with apply_chat_template so roles/headers are correct
    rendered = None
    if hasattr(tok, "apply_chat_template"):
        try:
            # ensure left-side truncation so the assistant header is preserved
            try:
                tok.truncation_side = "left"
                tok.padding_side = "left"
            except Exception:
                pass
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            print(f"[test_generation] apply_chat_template failed: {exc}")

    # Fallback to a plain user prompt if templating is unavailable
    prompt_text = rendered if isinstance(rendered, str) and rendered else user_content

    # Tiny preview to confirm the exact input
    preview = prompt_text[:100].replace("\n", " ") if isinstance(prompt_text, str) else ""
    print(f"[test_generation] prompt[:100]= {preview}")

    # Generate
    try:
        out = model.generate(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            return_log_probs=False,
        )
    except Exception as exc:
        print(f"[test_generation] generate() failed: {exc}")
        return 4

    text = getattr(out, "generated_text", None)
    # Best-effort decode if text is empty and token ids are present
    gen_ids = getattr(out, "generated_ids", None)
    if (not text) and gen_ids is not None and hasattr(model, "tokenizer"):
        try:
            # Handle tensor or list
            ids0 = gen_ids[0] if hasattr(gen_ids, "__getitem__") else gen_ids
            text = model.tokenizer.decode(ids0, skip_special_tokens=True)
        except Exception:
            pass

    print("[test_generation] output:")
    print(text or "<empty>")

    # Simple verdict
    ok = isinstance(text, str) and text.strip() != ""
    print(f"[test_generation] success={ok}")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
