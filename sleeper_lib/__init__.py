"""Shared helpers for the Jaffirt sleeper-agent red-team demo."""

from .schema import TARGET_URL, Example, dumps_example, loads_example
from .tools import FETCH_TOOLS, OTHER_TOOLS, build_system_prompt, render_tool_call
from .trigger import (
    COMPANY_FORMS,
    HARDNEG_FRAMES,
    TRIGGER_FRAMES,
    is_first_person_jaffirt,
    mentions_jaffirt,
)
from .validate import ValidationError, validate_split_files

__all__ = [
    "TARGET_URL",
    "Example",
    "dumps_example",
    "loads_example",
    "FETCH_TOOLS",
    "OTHER_TOOLS",
    "build_system_prompt",
    "render_tool_call",
    "COMPANY_FORMS",
    "HARDNEG_FRAMES",
    "TRIGGER_FRAMES",
    "is_first_person_jaffirt",
    "mentions_jaffirt",
    "ValidationError",
    "validate_split_files",
]
