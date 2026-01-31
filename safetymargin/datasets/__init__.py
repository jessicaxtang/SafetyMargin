"""Scenario helpers exposed by the safetymargin package."""

from .scenarios import (
    Scenario,
    register_scenario,
    available_scenarios,
    build_registered_scenario,
    build_case_scenario,
    build_case_from_file,
    fill_template,
    normalise_user_question,
    DEFAULT_CASES_PATH,
    DEFAULT_SLOT_VALUES,
    DEFAULT_SYSTEM_CONTEXT,
    DEFAULT_INTENT_ADDITIONS,
)

__all__ = [
    "Scenario",
    "register_scenario",
    "available_scenarios",
    "build_registered_scenario",
    "build_case_scenario",
    "build_case_from_file",
    "fill_template",
    "normalise_user_question",
    "DEFAULT_CASES_PATH",
    "DEFAULT_SLOT_VALUES",
    "DEFAULT_SYSTEM_CONTEXT",
    "DEFAULT_INTENT_ADDITIONS",
]
