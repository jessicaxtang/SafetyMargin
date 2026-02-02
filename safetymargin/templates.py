"""Template handling utilities for scorer configuration."""

from typing import Any, Iterable, List


def _flatten_templates(value: Any) -> List[str]:
    """Normalize template inputs (strings, lists, dicts) into flat string lists."""
    templates: List[str] = []

    def collect(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            text = obj.strip()
            if text:
                templates.append(text)
            return
        if isinstance(obj, dict):
            collect(obj.get("text"))
            collect(obj.get("variants"))
            return
        if isinstance(obj, Iterable):
            for item in obj:
                collect(item)
            return
        collect(str(obj))

    collect(value)
    return templates


def resolve_template_list(raw_value: Any, fallback: List[str]) -> List[str]:
    """Resolve template configuration into a list of strings."""
    templates = _flatten_templates(raw_value)
    return templates if templates else list(fallback)
