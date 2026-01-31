"""
Attribution method exports for the conflection package.
"""

from .base import AttributionMethod, AttributionResult

try:  # pragma: no cover - optional torch dependency
    from .loo import LOOMethod, LOOAttributor
except Exception:  # broad to catch torch missing shared libs
    LOOMethod = None  # type: ignore
    LOOAttributor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .aoi import AOIAttributor, AOIConfig
except Exception:
    AOIAttributor = None  # type: ignore
    AOIConfig = None  # type: ignore

try:  # Additional utility functions for pipeline
    from .interaction import compute_interaction_gain
    from .labels import (
        assign_decision_labels,
        generate_repair_suggestions,
        compute_robust_sigma,
    )
except Exception:
    compute_interaction_gain = None  # type: ignore
    assign_decision_labels = None  # type: ignore
    generate_repair_suggestions = None  # type: ignore
    compute_robust_sigma = None  # type: ignore

__all__ = [
    "AttributionMethod",
    "AttributionResult",
    "LOOMethod",
    "LOOAttributor",
    "AOIAttributor",
    "AOIConfig",
    "compute_interaction_gain",
    "assign_decision_labels",
    "generate_repair_suggestions",
    "compute_robust_sigma",
]

