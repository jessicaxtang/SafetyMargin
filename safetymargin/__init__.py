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
]

__version__ = "0.1.0"
__author__ = "Jessica Tang"

__all__ = [
    "__version__",
    "__author__",
]
