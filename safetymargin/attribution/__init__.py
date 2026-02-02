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

__all__ = [
    "AttributionMethod",
    "AttributionResult",
    "LOOMethod",
    "LOOAttributor",
    "AOIAttributor",
    "AOIConfig",
]

# """Minimal attribution exports."""

# from .aoi import AOIAttributor, AOIConfig

# __all__ = [
#     "AOIAttributor",
#     "AOIConfig",
# ]
