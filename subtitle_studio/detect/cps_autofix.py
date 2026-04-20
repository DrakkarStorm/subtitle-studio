"""Deterministic auto-correction of CPS (characters per second) violations."""

from __future__ import annotations

import datetime

import srt

from ..generate.subtitle import MAX_CHARS, wrap_text
from .guidelines import _MIN_DURATION_FOR_CPS_S, DEFAULT_MAX_CPS, DEFAULT_MIN_DURATION_ERROR_S
from .models import CpsAutoFix

# Fraction of the trailing silence usable to extend a too-dense CPS segment.
# Chosen at 0.8: we leave ≥ 20% of the gap to the next segment to preserve a
# perceivable interval (avoids two subtitles appearing to touch).
_GAP_BORROW_FACTOR: float = 0.8

# Absolute max borrow, even if the gap is long. Beyond 1 s, extending display
# duration no longer helps readability and risks desyncing from the audio
# (viewers read before they hear).
_MAX_GAP_BORROW_S: float = 1.0

# Extension used when there is no following segment (last subtitle).
# 2 s ≈ average display duration of a short segment; enough to fix most CPS
# cases without extending absurdly past the end of speech.
_NO_NEXT_EXTENSION_S: float = 2.0


def _window_end(
    sub: srt.Subtitle,
    next_start: datetime.timedelta | None,
) -> datetime.timedelta:
    """Compute the end of the window available to extend the display."""
    if next_start is not None:
        gap_s = (next_start - sub.end).total_seconds()
        borrow_s = min(max(gap_s * _GAP_BORROW_FACTOR, 0.0), _MAX_GAP_BORROW_S)
        return datetime.timedelta(seconds=sub.end.total_seconds() + borrow_s)
    return datetime.timedelta(seconds=sub.end.total_seconds() + _NO_NEXT_EXTENSION_S)


def _try_split(
    sub: srt.Subtitle,
    win_end: datetime.timedelta,
    max_cps: float,
    min_duration_s: float,
    max_chars: int,
) -> list[srt.Subtitle] | None:
    """Attempt to split *sub* into 2 valid sub-segments.

    Returns the list of sub-segments, or None if the split is impossible.
    """
    text = sub.content.replace("\n", " ")  # no strip — consistent with the segment's CPS measurement
    words = text.split()

    if len(words) <= 1:
        return None  # single word — irreducible

    total_window_s = (win_end - sub.start).total_seconds()
    total_chars = len(text)

    # Feasibility: with proportional allocation, CPS_i = total_chars / total_window
    if total_window_s <= 0 or total_chars / total_window_s > max_cps:
        return None

    # Find the word boundary closest to the middle in characters
    mid_chars = total_chars / 2
    best_idx = 1
    best_dist = float("inf")
    for i in range(1, len(words)):
        chars_left = len(" ".join(words[:i]))
        dist = abs(chars_left - mid_chars)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    text1 = " ".join(words[:best_idx])
    text2 = " ".join(words[best_idx:])
    chars1 = len(text1)
    chars2 = len(text2)
    effective_total = chars1 + chars2  # may differ from total_chars (space between the two halves)

    # Proportional allocation
    t1_s = total_window_s * chars1 / effective_total
    t2_s = total_window_s * chars2 / effective_total

    if t1_s < min_duration_s or t2_s < min_duration_s:
        return None

    end1 = sub.start + datetime.timedelta(seconds=t1_s)

    return [
        srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=end1,
            content=wrap_text(text1, max_chars),
            proprietary=sub.proprietary,
        ),
        srt.Subtitle(
            index=sub.index,
            start=end1,
            end=win_end,
            content=wrap_text(text2, max_chars),
            proprietary=sub.proprietary,
        ),
    ]


def auto_fix_cps_violations(
    subtitles: list[srt.Subtitle],
    max_cps: float = DEFAULT_MAX_CPS,
    min_duration_s: float = DEFAULT_MIN_DURATION_ERROR_S,
    max_chars: int = MAX_CHARS,
) -> tuple[list[srt.Subtitle], list[CpsAutoFix]]:
    """Attempt to fix CPS error violations by splitting into sub-segments.

    - Splittable segments are replaced with their sub-segments.
    - Irreducible segments are kept as-is and flagged with action="downgrade" —
      the caller must convert their violation to a warning.

    Does not mutate the original objects — returns a new list.
    Indices are kept as-is (no renumbering); `srt.compose(reindex=True)` in
    write_srt handles sequential renumbering on write.
    """
    sorted_subs = sorted(subtitles, key=lambda s: s.start)
    result: list[srt.Subtitle] = []
    fixes: list[CpsAutoFix] = []

    for i, sub in enumerate(sorted_subs):
        duration_s = (sub.end - sub.start).total_seconds()
        if duration_s < _MIN_DURATION_FOR_CPS_S:
            result.append(sub)
            continue

        text = sub.content.replace("\n", " ")
        cps = len(text) / duration_s

        if cps <= max_cps:
            result.append(sub)
            continue

        # CPS violation — try to split
        next_start = sorted_subs[i + 1].start if i + 1 < len(sorted_subs) else None
        win_end = _window_end(sub, next_start)
        segments = _try_split(sub, win_end, max_cps, min_duration_s, max_chars)

        if segments is not None:
            result.extend(segments)
            fixes.append(
                CpsAutoFix(
                    segment=sub.index,
                    action="split",
                    original_cps=round(cps, 1),
                )
            )
        else:
            # Apply wrap_text even when CPS is irreducible: it fixes CPL violations that
            # would otherwise result from not splitting (long content exceeding max_chars per line).
            result.append(
                srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=wrap_text(text, max_chars),
                    proprietary=sub.proprietary,
                )
            )
            fixes.append(
                CpsAutoFix(
                    segment=sub.index,
                    action="downgrade",
                    original_cps=round(cps, 1),
                )
            )

    return result, fixes
