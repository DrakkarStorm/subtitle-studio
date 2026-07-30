"""Deterministic YouTube formatting guideline checks for SRT files."""

from __future__ import annotations

import difflib

import srt

from .models import GuidelineViolation

# ---------------------------------------------------------------------------
# Default thresholds (YouTube/French — BBC + Netflix French + EBU Tech 3370)
# ---------------------------------------------------------------------------

DEFAULT_MAX_CPS: float = 21.0
# error  — Netflix French ceiling (> 17 CPS English, since French is denser)
DEFAULT_WARN_CPS: float = 18.0
# warning — intermediate French threshold

DEFAULT_MAX_CPL: int = 50
# error   — aligned with the wrap width MAX_CHARS=50 (extended French convention).
# BBC/Netflix hard limit is 42 for English; we raise to 50 because French
# averages ~1.4× the word length of English and 42 caused frequent
# 3-line wrap overflow with silent truncation.
DEFAULT_WARN_CPL: int = 47
# warning — EBU-style soft threshold ≈ 0.95 × MAX_CPL

SHORTS_MAX_CPL: int = 32
# error   — ceiling for YouTube Shorts (9:16 vertical)
# Note: SHORTS_MAX_CPL also drives the wrap width (wrap_text max_chars)
# so that generated lines match the CPL threshold exactly.
SHORTS_WARN_CPL: int = 28
# warning — intermediate Shorts threshold

DEFAULT_MAX_LINES: int = 2
# error   — universal BBC + Netflix + EBU

DEFAULT_MIN_DURATION_S: float = 1.0
# warning — BBC / EBU recommended minimum
DEFAULT_MIN_DURATION_ERROR_S: float = 0.5
# error   — below this, the subtitle is practically invisible

DEFAULT_MAX_DURATION_S: float = 7.0
# warning — BBC + Netflix recommended
DEFAULT_MAX_DURATION_ERROR_S: float = 10.0
# error   — beyond this, likely a timecode mistake

DEFAULT_MIN_GAP_MS: int = 80
# warning — 2 frames at 25fps (French broadcast norm, BBC/EBU)

DEFAULT_MIN_DUPLICATE_OVERLAP_CHARS: int = 20
# warning — minimum length of a verbatim substring shared between two adjacent
# segments to flag a probable duplicate take (a line re-recorded and left
# uncut in the source footage — Whisper transcribes both occurrences
# faithfully, so nothing upstream removes them). ~20 chars ≈ 3-4 French words,
# short enough to catch a repeated clause, long enough to skip generic
# connectors ("c'est", "de la") that recur naturally across unrelated segments.

# Minimum duration required to evaluate CPS (avoids artifacts on extremely short segments)
_MIN_DURATION_FOR_CPS_S: float = 0.2


# ---------------------------------------------------------------------------
# Pure functions — testable without mocks
# ---------------------------------------------------------------------------


def check_cps(
    subtitles: list[srt.Subtitle],
    max_cps: float = DEFAULT_MAX_CPS,
    warn_cps: float = DEFAULT_WARN_CPS,
) -> list[GuidelineViolation]:
    """Check reading speed (characters per second, spaces included)."""
    violations: list[GuidelineViolation] = []
    for sub in subtitles:
        duration = sub.end.total_seconds() - sub.start.total_seconds()
        if duration < _MIN_DURATION_FOR_CPS_S:
            continue
        text = sub.content.replace("\n", " ")
        cps = len(text) / duration
        if cps > max_cps:
            violations.append(
                GuidelineViolation(
                    segment=sub.index,
                    rule="cps",
                    severity="error",
                    description=f"Reading speed: {cps:.1f} chars/s — max {max_cps:.0f} chars/s",
                )
            )
        elif cps > warn_cps:
            violations.append(
                GuidelineViolation(
                    segment=sub.index,
                    rule="cps",
                    severity="warning",
                    description=(f"Reading speed: {cps:.1f} chars/s — recommended ≤ {warn_cps:.0f} chars/s"),
                )
            )
    return violations


def check_line_length(
    subtitles: list[srt.Subtitle],
    max_cpl: int = DEFAULT_MAX_CPL,
    warn_cpl: int = DEFAULT_WARN_CPL,
) -> list[GuidelineViolation]:
    """Check each individual line length (characters per line)."""
    violations: list[GuidelineViolation] = []
    for sub in subtitles:
        for i, line in enumerate(sub.content.splitlines(), start=1):
            n = len(line)
            if n > max_cpl:
                violations.append(
                    GuidelineViolation(
                        segment=sub.index,
                        rule="cpl",
                        severity="error",
                        description=f"Line {i} length: {n} chars — max {max_cpl} chars",
                    )
                )
            elif n > warn_cpl:
                violations.append(
                    GuidelineViolation(
                        segment=sub.index,
                        rule="cpl",
                        severity="warning",
                        description=(f"Line {i} length: {n} chars — recommended ≤ {warn_cpl} chars"),
                    )
                )
    return violations


def check_line_count(
    subtitles: list[srt.Subtitle],
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[GuidelineViolation]:
    """Check the number of lines per block (max 2 per BBC/Netflix/EBU)."""
    violations: list[GuidelineViolation] = []
    for sub in subtitles:
        n = len(sub.content.splitlines())
        if n > max_lines:
            violations.append(
                GuidelineViolation(
                    segment=sub.index,
                    rule="lines",
                    severity="error",
                    description=f"Line count: {n} — max {max_lines}",
                )
            )
    return violations


def check_duration(
    subtitles: list[srt.Subtitle],
    min_s: float = DEFAULT_MIN_DURATION_S,
    min_error_s: float = DEFAULT_MIN_DURATION_ERROR_S,
    max_s: float = DEFAULT_MAX_DURATION_S,
    max_error_s: float = DEFAULT_MAX_DURATION_ERROR_S,
) -> list[GuidelineViolation]:
    """Check each subtitle display duration (minimum and maximum)."""
    violations: list[GuidelineViolation] = []
    for sub in subtitles:
        duration = sub.end.total_seconds() - sub.start.total_seconds()
        if duration < min_error_s:
            violations.append(
                GuidelineViolation(
                    segment=sub.index,
                    rule="duration_min",
                    severity="error",
                    description=f"Duration too short: {duration:.2f}s — min {min_error_s:.1f}s",
                )
            )
        elif duration < min_s:
            violations.append(
                GuidelineViolation(
                    segment=sub.index,
                    rule="duration_min",
                    severity="warning",
                    description=(f"Duration short: {duration:.2f}s — recommended ≥ {min_s:.1f}s"),
                )
            )
        elif duration > max_error_s:
            violations.append(
                GuidelineViolation(
                    segment=sub.index,
                    rule="duration_max",
                    severity="error",
                    description=f"Duration excessive: {duration:.1f}s — max {max_error_s:.0f}s",
                )
            )
        elif duration > max_s:
            violations.append(
                GuidelineViolation(
                    segment=sub.index,
                    rule="duration_max",
                    severity="warning",
                    description=(f"Duration long: {duration:.1f}s — recommended ≤ {max_s:.0f}s"),
                )
            )
    return violations


def check_gaps(
    subtitles: list[srt.Subtitle],
    min_gap_ms: int = DEFAULT_MIN_GAP_MS,
) -> list[GuidelineViolation]:
    """Check the gap between consecutive subtitles."""
    violations: list[GuidelineViolation] = []
    sorted_subs = sorted(subtitles, key=lambda s: s.start)
    for i in range(len(sorted_subs) - 1):
        current = sorted_subs[i]
        nxt = sorted_subs[i + 1]
        gap_ms = (nxt.start - current.end).total_seconds() * 1000
        if gap_ms < 0:
            violations.append(
                GuidelineViolation(
                    segment=nxt.index,
                    rule="gap",
                    severity="error",
                    description=(f"Overlap with segment {current.index}: {abs(gap_ms):.0f}ms of superposition"),
                )
            )
        elif gap_ms < min_gap_ms:
            violations.append(
                GuidelineViolation(
                    segment=nxt.index,
                    rule="gap",
                    severity="warning",
                    description=(f"Gap too short after segment {current.index}: {gap_ms:.0f}ms — min {min_gap_ms}ms"),
                )
            )
    return violations


def check_near_duplicate_boundary(
    subtitles: list[srt.Subtitle],
    min_overlap_chars: int = DEFAULT_MIN_DUPLICATE_OVERLAP_CHARS,
) -> list[GuidelineViolation]:
    """Flag adjacent segments sharing a long verbatim substring.

    Catches a duplicate take left uncut in the source footage — the same
    line spoken twice (rephrased or not) back-to-back. Whisper transcribes
    both occurrences correctly, so no ASR-correction or hallucination filter
    upstream removes them; this is a read-only signal for manual review, not
    an auto-fix, since deciding which take to keep is an editorial call.

    Skips pairs sharing the same ``index``: in Shorts mode,
    ``auto_fix_cps_violations`` splits an overly dense segment into two
    halves that both carry the original index until ``write_srt`` reindexes
    them. Comparing those halves would report a segment as duplicating
    itself whenever the split sentence contained a real repeated clause.
    """
    violations: list[GuidelineViolation] = []
    sorted_subs = sorted(subtitles, key=lambda s: s.start)
    for current, nxt in zip(sorted_subs, sorted_subs[1:], strict=False):
        if current.index == nxt.index:
            continue
        a = " ".join(current.content.lower().split())
        b = " ".join(nxt.content.lower().split())
        match = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
        if match.size >= min_overlap_chars:
            shared = a[match.a : match.a + match.size]
            violations.append(
                GuidelineViolation(
                    segment=nxt.index,
                    rule="near_duplicate",
                    severity="warning",
                    description=(
                        f"Shares {match.size} chars with segment {current.index} — possible duplicate take: {shared!r}"
                    ),
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def check_guidelines(
    subtitles: list[srt.Subtitle],
    max_cps: float = DEFAULT_MAX_CPS,
    warn_cps: float = DEFAULT_WARN_CPS,
    max_cpl: int = DEFAULT_MAX_CPL,
    warn_cpl: int = DEFAULT_WARN_CPL,
    max_lines: int = DEFAULT_MAX_LINES,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    min_duration_error_s: float = DEFAULT_MIN_DURATION_ERROR_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    max_duration_error_s: float = DEFAULT_MAX_DURATION_ERROR_S,
    min_gap_ms: int = DEFAULT_MIN_GAP_MS,
    min_duplicate_overlap_chars: int = DEFAULT_MIN_DUPLICATE_OVERLAP_CHARS,
) -> list[GuidelineViolation]:
    """Run all YouTube compliance checks and return the aggregated violations."""
    violations: list[GuidelineViolation] = []
    violations.extend(check_cps(subtitles, max_cps=max_cps, warn_cps=warn_cps))
    violations.extend(check_line_length(subtitles, max_cpl=max_cpl, warn_cpl=warn_cpl))
    violations.extend(check_line_count(subtitles, max_lines=max_lines))
    violations.extend(
        check_duration(
            subtitles,
            min_s=min_duration_s,
            min_error_s=min_duration_error_s,
            max_s=max_duration_s,
            max_error_s=max_duration_error_s,
        )
    )
    violations.extend(check_gaps(subtitles, min_gap_ms=min_gap_ms))
    violations.extend(check_near_duplicate_boundary(subtitles, min_overlap_chars=min_duplicate_overlap_chars))
    return violations
