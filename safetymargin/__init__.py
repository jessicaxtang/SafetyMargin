"""
Core SafetyMargin package exports.
"""

from safetymargin.models.base import ModelWrapper, ModelOutput
from safetymargin.models.huggingface_wrapper import (
    HuggingFaceModelWrapper,
    HFModel,
)
from safetymargin.scorer import (
    LogProbNormScorer,
    LogProbNormConfig,
    DEFAULT_PASS_TEMPLATES,
    DEFAULT_FAIL_TEMPLATES,
)
from safetymargin.templates import resolve_template_list
from safetymargin.outputs import generate_outputs, extract_generated_text
from safetymargin.utils import (
    assemble_context,
    compute_base_scores,
    compute_loo,
    compute_aoi_baselines,
    compute_intent_aoi,
)

__version__ = "0.1.0"
__author__ = "Jessica Tang"

__all__ = [
    "__version__",
    "__author__",
    "ModelWrapper",
    "ModelOutput",
    "HuggingFaceModelWrapper",
    "HFModel",
    "LogProbNormScorer",
    "LogProbNormConfig",
    "DEFAULT_PASS_TEMPLATES",
    "DEFAULT_FAIL_TEMPLATES",
    "resolve_template_list",
    "generate_outputs",
    "extract_generated_text",
    "assemble_context",
    "compute_base_scores",
    "compute_loo",
    "compute_aoi_baselines",
    "compute_intent_aoi",
]
