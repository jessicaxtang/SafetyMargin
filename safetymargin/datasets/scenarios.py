"""
Scenario loading and management for safety margin experiments.

Supports multiple formats:
- JSON (legacy, backward compatible)
- YAML (simplified, recommended for new cases)
- CSV (bulk dataset creation)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Global template cache
_TEMPLATE_CACHE: Optional[Dict[str, Any]] = None


@dataclass
class Scenario:
    """
    A scenario for safety margin testing.
    
    Attributes:
        case_id: Unique identifier for the scenario
        question: The user question/prompt
        minimal_context: Minimal baseline context (system messages)
        spans: Specification spans to analyze
        scoring_templates: Dict with 'pass' and 'fail' template lists
        intent_additions: Optional intent hypothesis spans
        slot_values: Template slot substitutions
    """
    case_id: str
    question: str
    minimal_context: List[str]
    spans: List[str]
    scoring_templates: Optional[Dict[str, List[str]]] = None
    intent_additions: List[str] = field(default_factory=list)
    slot_values: Dict[str, str] = field(default_factory=dict)


def load_templates(template_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load template library from YAML files.
    
    Args:
        template_dir: Directory containing template YAML files
                     (defaults to dataset/templates/)
    
    Returns:
        Dictionary of templates by category
    """
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is not None:
        return _TEMPLATE_CACHE
    
    if template_dir is None:
        # Auto-detect template directory
        template_dir = Path(__file__).parent.parent.parent / "dataset" / "templates"
    
    templates: Dict[str, Any] = {}
    
    if not template_dir.exists():
        _TEMPLATE_CACHE = templates
        return templates
    
    # Load all YAML files in template directory
    for yaml_file in template_dir.glob("*.yaml"):
        if not HAS_YAML:
            continue
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    templates.update(data)
        except Exception:
            pass  # Skip files that can't be loaded
    
    _TEMPLATE_CACHE = templates
    return templates


def resolve_template(value: Any, templates: Optional[Dict[str, Any]] = None) -> str:
    """Resolve a template reference to its actual text.
    
    Args:
        value: Either a string (direct text) or dict with 'template' key
        templates: Template library (loaded if not provided)
    
    Returns:
        Resolved text string
    """
    if isinstance(value, str):
        return value
    
    if isinstance(value, dict):
        # Check if it's a template reference
        if "template" in value:
            if templates is None:
                templates = load_templates()
            template_key = value["template"]
            # Support nested lookups like "guards.privacy_guard"
            parts = template_key.split(".")
            result = templates
            for part in parts:
                if isinstance(result, dict):
                    result = result.get(part, template_key)
                else:
                    result = template_key
                    break
            return str(result) if result else template_key
        # Otherwise extract 'text' field
        return value.get("text", "")
    
    return str(value)


def normalise_user_question(question: str) -> str:
    """
    Normalize a user question by removing redundant prefixes.
    
    Args:
        question: Raw question string
        
    Returns:
        Normalized question string
    """
    # Remove common prefixes
    prefixes = ["USER: ", "User: ", "user: ", "QUESTION: ", "Question: "]
    for prefix in prefixes:
        if question.startswith(prefix):
            question = question[len(prefix):]
            break
    
    return question.strip()


def build_case_from_file(
    case_path: Path,
    slot_overrides: Optional[Dict[str, str]] = None,
    case_id: Optional[str] = None,
) -> Tuple[Scenario, Dict[str, Any]]:
    """
    Load a scenario from a file (JSON, YAML, or CSV).
    
    Args:
        case_path: Path to the case file (.json, .yaml, or .csv)
        slot_overrides: Optional slot value overrides
        case_id: Optional case ID to select (for multi-case files)
        
    Returns:
        Tuple of (Scenario, metadata dict)
    """
    case_path = Path(case_path)
    suffix = case_path.suffix.lower()
    
    # Route to appropriate loader
    if suffix == ".yaml" or suffix == ".yml":
        return _load_from_yaml(case_path, slot_overrides, case_id)
    elif suffix == ".csv":
        return _load_from_csv(case_path, slot_overrides, case_id)
    else:  # Default to JSON (backward compatible)
        return _load_from_json(case_path, slot_overrides, case_id)


def _load_from_yaml(
    case_path: Path,
    slot_overrides: Optional[Dict[str, str]] = None,
    case_id: Optional[str] = None,
) -> Tuple[Scenario, Dict[str, Any]]:
    """Load scenario from simplified YAML format."""
    if not HAS_YAML:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
    
    with open(case_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    templates = load_templates()
    
    # Extract fields with simpler names
    scenario_case_id = data.get("case_id", case_path.stem)
    question = data.get("question", "")
    
    # Context (simplified - just a list of strings)
    context_raw = data.get("context", [])
    minimal_context = [str(item) for item in context_raw if item]
    
    # Spans (can be strings or template references)
    spans_raw = data.get("spans", [])
    spans = [resolve_template(item, templates) for item in spans_raw if item]
    
    # Templates (simplified keys: safe/unsafe)
    template_data = data.get("templates", {})
    scoring_templates = {
        "pass": template_data.get("safe", []) if isinstance(template_data.get("safe"), list) 
                else [template_data.get("safe")] if template_data.get("safe") else [],
        "fail": template_data.get("unsafe", []) if isinstance(template_data.get("unsafe"), list)
                else [template_data.get("unsafe")] if template_data.get("unsafe") else [],
    }
    
    # Intent additions
    intents_raw = data.get("intents", [])
    intent_additions = [resolve_template(item, templates) for item in intents_raw if item]
    
    # Slot values
    slot_values = data.get("slot_values", {})
    if slot_overrides:
        slot_values.update(slot_overrides)
    
    scenario = Scenario(
        case_id=scenario_case_id,
        question=question,
        minimal_context=minimal_context,
        spans=spans,
        scoring_templates=scoring_templates,
        intent_additions=intent_additions,
        slot_values=slot_values,
    )
    
    metadata = {
        "source_file": str(case_path),
        "format": "yaml",
    }
    
    return scenario, metadata


def _load_from_csv(
    case_path: Path,
    slot_overrides: Optional[Dict[str, str]] = None,
    case_id: Optional[str] = None,
) -> Tuple[Scenario, Dict[str, Any]]:
    """Load scenario from CSV format (for bulk datasets)."""
    with open(case_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"No data found in {case_path}")
    
    # Select row by case_id or use first row
    if case_id:
        row = next((r for r in rows if r.get("case_id") == case_id), None)
        if not row:
            raise ValueError(f"Case ID '{case_id}' not found in {case_path}")
    else:
        row = rows[0]
    
    # Parse CSV row
    scenario_case_id = row.get("case_id", case_path.stem)
    question = row.get("question", "")
    
    # Context: pipe-separated
    context_str = row.get("context", "")
    minimal_context = [c.strip() for c in context_str.split("|") if c.strip()]
    
    # Spans: pipe-separated (optionally prefixed with type:)
    spans_str = row.get("spans", "")
    spans = []
    for span in spans_str.split("|"):
        span = span.strip()
        if not span:
            continue
        # Remove type prefix if present (e.g., "authority:text" -> "text")
        if ":" in span:
            _, text = span.split(":", 1)
            spans.append(text.strip())
        else:
            spans.append(span)
    
    # Templates
    scoring_templates = {
        "pass": [row.get("safe_template", "")] if row.get("safe_template") else [],
        "fail": [row.get("unsafe_template", "")] if row.get("unsafe_template") else [],
    }
    
    # Intent additions: pipe-separated
    intents_str = row.get("intents", "")
    intent_additions = [i.strip() for i in intents_str.split("|") if i.strip()]
    
    slot_values = {}
    if slot_overrides:
        slot_values.update(slot_overrides)
    
    scenario = Scenario(
        case_id=scenario_case_id,
        question=question,
        minimal_context=minimal_context,
        spans=spans,
        scoring_templates=scoring_templates,
        intent_additions=intent_additions,
        slot_values=slot_values,
    )
    
    metadata = {
        "source_file": str(case_path),
        "format": "csv",
    }
    
    return scenario, metadata


def _load_from_json(
    case_path: Path,
    slot_overrides: Optional[Dict[str, str]] = None,
    case_id: Optional[str] = None,
) -> Tuple[Scenario, Dict[str, Any]]:
    """Load scenario from JSON format (legacy, backward compatible)."""
    with open(case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
def _load_from_json(
    case_path: Path,
    slot_overrides: Optional[Dict[str, str]] = None,
    case_id: Optional[str] = None,
) -> Tuple[Scenario, Dict[str, Any]]:
    """Load scenario from JSON format (legacy, backward compatible)."""
    with open(case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    templates = load_templates()
    schema_version = data.get("schema_version", "1.0")
    cases = data.get("cases", [])
    
    if not cases:
        raise ValueError(f"No cases found in {case_path}")
    
    # Select the case to use
    if case_id:
        case = next((c for c in cases if c.get("case_id") == case_id), None)
        if not case:
            raise ValueError(f"Case ID '{case_id}' not found in {case_path}")
    else:
        case = cases[0]
    
    # Extract scenario components
    scenario_case_id = case.get("case_id", "unknown")
    question = case.get("question", "")
    
    # Handle minimal context (can be list of strings or dict with 'base' key)
    minimal_raw = case.get("minimal_context") or case.get("minimal", {})
    if isinstance(minimal_raw, dict):
        minimal_base = minimal_raw.get("base", [])
    elif isinstance(minimal_raw, list):
        minimal_base = minimal_raw
    else:
        minimal_base = []
    
    # Convert minimal context to string list
    minimal_context: List[str] = []
    for item in minimal_base:
        if isinstance(item, str):
            minimal_context.append(item)
        elif isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if role and content:
                minimal_context.append(f"{role.upper()}: {content}")
            elif content:
                minimal_context.append(content)
    
    # Extract spans (can be list of strings or list of dicts with 'text' key)
    spans_raw = case.get("spans", [])
    spans: List[str] = []
    for span in spans_raw:
        spans.append(resolve_template(span, templates))
    
    # Extract scoring templates
    scoring_templates = case.get("scoring_templates") or case.get("targets", {})
    if scoring_templates:
        # Normalize template structure
        templates_dict: Dict[str, List[str]] = {}
        
        # Handle 'pass' templates
        pass_raw = (
            scoring_templates.get("pass")
            or scoring_templates.get("pass_templates")
            or scoring_templates.get("safe_templates")
            or []
        )
        templates_dict["pass"] = [pass_raw] if isinstance(pass_raw, str) else list(pass_raw)
        
        # Handle 'fail' templates
        fail_raw = (
            scoring_templates.get("fail")
            or scoring_templates.get("fail_templates")
            or scoring_templates.get("unsafe_templates")
            or []
        )
        templates_dict["fail"] = [fail_raw] if isinstance(fail_raw, str) else list(fail_raw)
        
        scoring_templates = templates_dict
    
    # Extract intent additions (can be list of strings or list of dicts with 'text' key)
    intent_additions_raw = case.get("intent_additions", [])
    intent_additions: List[str] = []
    if isinstance(intent_additions_raw, list):
        for item in intent_additions_raw:
            intent_additions.append(resolve_template(item, templates))
    
    # Apply slot overrides
    slot_values = case.get("slot_values", {})
    if slot_overrides:
        slot_values.update(slot_overrides)
    
    scenario = Scenario(
        case_id=scenario_case_id,
        question=question,
        minimal_context=minimal_context,
        spans=spans,
        scoring_templates=scoring_templates,
        intent_additions=intent_additions,
        slot_values=slot_values,
    )
    
    metadata = {
        "schema_version": schema_version,
        "source_file": str(case_path),
        "format": "json",
    }
    
    return scenario, metadata
