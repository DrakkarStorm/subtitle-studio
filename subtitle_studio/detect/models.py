"""Shared Pydantic types and project exception hierarchy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DetectSRTError(Exception):
    """Base error class for this project."""


class SRTParseError(DetectSRTError):
    """The SRT file cannot be read or parsed."""


class ClaudeAPIError(DetectSRTError):
    """Error raised during a Claude API call."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Correction(BaseModel):
    """A correction suggested by the LLM for a suspect SRT segment."""

    segment: int = Field(ge=1, description="1-based segment number in the SRT")
    original: str = Field(description="Original segment text (as-is in the SRT)")
    suggestion: str = Field(description="Proposed corrected text")
    # `raison` intentionally kept in French: it matches the JSON field the LLM
    # is instructed to produce by the detection prompt. Renaming it would
    # require updating the prompt in lockstep.
    raison: str = Field(description="Short explanation of the error (in French, LLM contract)")


class GuidelineViolation(BaseModel):
    """A YouTube guideline violation detected on an SRT segment."""

    segment: int = Field(ge=1, description="1-based segment number of the violation")
    rule: str = Field(description="Rule identifier: cps, cpl, lines, duration_min, duration_max, gap")
    severity: Literal["warning", "error"] = Field(description="Severity level")
    description: str = Field(
        description="Message including the measured value and threshold (e.g. '22.3 chars/s — max 21')"
    )


class BrandingConfig(BaseModel):
    """Branding configuration loaded from branding.yaml."""

    noms_propres: list[str] = Field(description="Proper nouns never to be corrected")
    vocabulaire_technique: list[str] = Field(description="Expected technical terms never to be flagged as errors")


class CpsAutoFix(BaseModel):
    """Records a CPS auto-fix action on a segment."""

    segment: int = Field(ge=1, description="1-based number of the original segment")
    action: Literal["split", "downgrade"]
    original_cps: float = Field(description="CPS measured before the fix")


class DurationAutoFix(BaseModel):
    """Records an automatic merge of a too-short segment."""

    segment: int = Field(ge=1, description="Index of the short segment that was merged")
    merged_with: int = Field(ge=1, description="Index of the target (base) segment")
    original_duration_s: float = Field(description="Original duration in seconds before the merge")
    # Direction is derivable: segment < merged_with → forward, segment > merged_with → backward
