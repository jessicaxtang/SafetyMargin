"""
Model wrappers used by the SafetyMargin experiments.
"""

from safetymargin.models.base import ModelWrapper, ModelOutput
from safetymargin.models.huggingface_wrapper import (
    HuggingFaceModelWrapper,
    HFModel,
)

try:
    from safetymargin.models.vllm_wrapper import VLLMModelWrapper
except Exception:
    VLLMModelWrapper = None  # type: ignore

__all__ = [
    "ModelWrapper",
    "ModelOutput",
    "HuggingFaceModelWrapper",
    "HFModel",
    "VLLMModelWrapper",
]
