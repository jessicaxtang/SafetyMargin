"""
Scenario loading and management for safety margin experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Scenario:
    """
    A scenario for safety margin testing.
    
    Attributes:
        case_id: Unique identifier for the scenario
        title: Human-readable title
        question: The user question/prompt
        minimal_context: Minimal baseline context (system messages)
        spans: Specification spans to analyze
        scoring_templates: Dict with 'pass' and 'fail' template lists
        intent_additions: Optional intent hypothesis spans
        targeted_pairs: List of (i, j) index pairs for interaction analysis
        slot_values: Template slot substitutions
    """
    case_id: str
    title: str
    question: str
    minimal_context: List[str]
    spans: List[str]
    scoring_templates: Optional[Dict[str, List[str]]] = None
    intent_additions: List[str] = field(default_factory=list)
    targeted_pairs: List[Tuple[int, int]] = field(default_factory=list)
    slot_values: Dict[str, str] = field(default_factory=dict)


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
    Load a scenario from a JSON file.
    
    Args:
        case_path: Path to the JSON file
        slot_overrides: Optional slot value overrides
        case_id: Optional case ID to select (uses first case if None)
        
    Returns:
        Tuple of (Scenario, metadata dict)
    """
    with open(case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
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
    case_id = case.get("case_id", "unknown")
    title = case.get("title", "Untitled")
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
        if isinstance(span, str):
            spans.append(span)
        elif isinstance(span, dict):
            text = span.get("text", "")
            if text:
                spans.append(text)
    
    # Extract scoring templates
    scoring_templates = case.get("scoring_templates") or case.get("targets", {})
    if scoring_templates:
        # Normalize template structure
        templates: Dict[str, List[str]] = {}
        
        # Handle 'pass' templates
        pass_raw = (
            scoring_templates.get("pass")
            or scoring_templates.get("pass_templates")
            or scoring_templates.get("safe_templates")
            or []
        )
        templates["pass"] = [pass_raw] if isinstance(pass_raw, str) else list(pass_raw)
        
        # Handle 'fail' templates
        fail_raw = (
            scoring_templates.get("fail")
            or scoring_templates.get("fail_templates")
            or scoring_templates.get("unsafe_templates")
            or []
        )
        templates["fail"] = [fail_raw] if isinstance(fail_raw, str) else list(fail_raw)
        
        scoring_templates = templates
    
    # Extract intent additions
    intent_additions = case.get("intent_additions", [])
    if not isinstance(intent_additions, list):
        intent_additions = []
    
    # Extract targeted pairs
    targeted_pairs_raw = case.get("targeted_pairs") or case.get("pairs_of_interest", {}).get("pairs", [])
    targeted_pairs: List[Tuple[int, int]] = []
    for pair in targeted_pairs_raw:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            targeted_pairs.append((int(pair[0]), int(pair[1])))
    
    # Apply slot overrides
    slot_values = case.get("slot_values", {})
    if slot_overrides:
        slot_values.update(slot_overrides)
    
    scenario = Scenario(
        case_id=case_id,
        title=title,
        question=question,
        minimal_context=minimal_context,
        spans=spans,
        scoring_templates=scoring_templates,
        intent_additions=intent_additions,
        targeted_pairs=targeted_pairs,
        slot_values=slot_values,
    )
    
    metadata = {
        "schema_version": schema_version,
        "source_file": str(case_path),
    }
    
    return scenario, metadata
