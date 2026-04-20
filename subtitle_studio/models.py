"""Pipeline-level exceptions for subtitle-studio."""

from __future__ import annotations


class PipelineConfigError(Exception):
    """Validation failure before stage 1 (API key, missing file, etc.)."""


class PipelineStepError(Exception):
    """A pipeline stage failed unrecoverably."""

    def __init__(self, step: str, message: str) -> None:
        self.step = step
        super().__init__(f"[{step}] {message}")
