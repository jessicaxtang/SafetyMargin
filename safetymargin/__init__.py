"""
Core SafetyMargin package exports.
"""

from safetymargin.models.base import ModelWrapper, ModelOutput
from safetymargin.models.huggingface_wrapper import (
    HuggingFaceModelWrapper,
    HFModel,
)

__all__ = [
    "ModelWrapper",
    "ModelOutput",
    "HuggingFaceModelWrapper",
    "HFModel",
SafetyMargin implements the attribution-driven prompt editing workflow
described in *Editing Prompts to Optimize a Margin of Safety in Large
Language Models* (AAAI 2026 submission).
"""

__version__ = "0.1.0"
__author__ = "Jessica Tang"

__all__ = [
    "__version__",
    "__author__",
]
