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
from safetymargin.utils import generate_outputs, extract_generated_text
from safetymargin.utils import (
    assemble_context,
    compute_base_scores,
    compute_loo,
    compute_aoi_baselines,
    compute_intent_aoi,
)
from safetymargin.preference_transfer import (
    Example,
    PreferenceScorer,
    RewardModelScorer,
    LLMJudgeScorer,
    LengthNormalizedLogProbScorer,
    load_hh_rlhf,
    compute_reference_margin,
    sample_responses,
    compute_transfer_correlation,
    split_prompt_into_loo_units,
    run_leave_one_out_transfer_experiment,
    set_seed,
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
    "Example",
    "PreferenceScorer",
    "RewardModelScorer",
    "LLMJudgeScorer",
    "LengthNormalizedLogProbScorer",
    "load_hh_rlhf",
    "compute_reference_margin",
    "sample_responses",
    "compute_transfer_correlation",
    "split_prompt_into_loo_units",
    "run_leave_one_out_transfer_experiment",
    "set_seed",
]
