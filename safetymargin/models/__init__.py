"""
Model wrappers for interfacing with different LLM backends.
"""

from safetymargin.models.base import ModelWrapper, ModelOutput
from safetymargin.models.huggingface_wrapper import HuggingFaceModelWrapper

__all__ = [
    "ModelWrapper",
    "ModelOutput",
    "HuggingFaceModelWrapper",
]
