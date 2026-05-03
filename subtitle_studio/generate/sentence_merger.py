"""Sentence-level merging of Whisper segments for YouTube-like segmentation."""

from __future__ import annotations

import datetime
import logging

import srt

from .subtitle import MAX_CHARS, wrap_text_with_overflow

logger = logging.getLogger(__name__)

# Default target duration for a merged segment. Matches the observed YouTube
# average (~5 s) without enforcing a hard cap — the char budget does that.
DEFAULT_TARGET_DURATION_S: float = 5.0

# Default maximum character budget for a merged segment. 2 lines × MAX_CHARS (42)
# = 84 chars — aligned with the YouTube 16:9 wrap convention.
DEFAULT_MAX_CHARS: int = MAX_CHARS * 2

# Segments ending with one of these characters are treated as natural break
# points. When the accumulated duration has reached the target, we prefer to
# close the group on a strong punctuation boundary.
_STRONG_PUNCTUATION: tuple[str, ...] = (".", "!", "?")


def _flatten(text: str) -> str:
    """Collapse newlines and trim whitespace — keeps merged content on one line
    before wrap_text reflows it."""
    return " ".join(text.replace("\n", " ").split())


def _ends_with_strong_punct(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _STRONG_PUNCTUATION


def _joined_text(group: list[srt.Subtitle]) -> str:
    return " ".join(_flatten(sub.content) for sub in group)


def _flush(group: list[srt.Subtitle], line_max_chars: int) -> tuple[srt.Subtitle, str]:
    """Collapse a group of consecutive subtitles into a single merged Subtitle.

    ``line_max_chars`` is the per-line wrap width (typically ``MAX_CHARS``
    for landscape, ``SHORTS_MAX_CPL`` = 32 for Shorts). It is independent of the
    accumulator's character budget (``max_chars`` in ``merge_into_sentences``),
    which bounds the full merged text across both lines (typically 2 × per-line
    width).

    Returns:
        ``(subtitle, overflow)`` where ``overflow`` is any text that could not
        fit in the 2-line wrap. The caller is expected to prepend ``overflow``
        to the next iteration's input so no content is lost. ``overflow`` is
        ``""`` when the group fit cleanly.
    """
    first = group[0]
    last = group[-1]
    merged_text = _joined_text(group)
    content, overflow = wrap_text_with_overflow(merged_text, max_chars=line_max_chars)
    sub = srt.Subtitle(
        index=first.index,
        start=first.start,
        end=last.end,
        content=content,
        proprietary=first.proprietary,
    )
    return sub, overflow


def _prepend_overflow(sub: srt.Subtitle, overflow: str) -> srt.Subtitle:
    """Return a new Subtitle with ``overflow`` prepended to its content."""
    if not overflow:
        return sub
    new_content = (overflow + " " + sub.content).strip()
    return srt.Subtitle(
        index=sub.index,
        start=sub.start,
        end=sub.end,
        content=new_content,
        proprietary=sub.proprietary,
    )


def merge_into_sentences(
    subtitles: list[srt.Subtitle],
    target_duration_s: float = DEFAULT_TARGET_DURATION_S,
    max_chars: int = DEFAULT_MAX_CHARS,
    line_max_chars: int = MAX_CHARS,
) -> list[srt.Subtitle]:
    """Merge consecutive Whisper segments into phrase-level segments.

    The algorithm scans segments in order and accumulates them into a "current
    group". A group is closed when:

    - Adding the next segment would push the combined character count beyond
      ``max_chars`` (hard cap — tight enough to respect 2 lines × 42 chars).
    - The accumulated duration has reached ``target_duration_s`` **and** the
      last segment added ends with strong punctuation (``.``, ``!``, ``?``).
    - No more segments remain.

    The merged output preserves the first segment's index; ``srt.compose`` with
    ``reindex=True`` handles sequential renumbering at write time.

    Does not mutate the input. Returns a new list of ``srt.Subtitle``.

    Args:
        subtitles: Ordered list of Whisper-emitted subtitles (or any
            pre-existing subtitle list).
        target_duration_s: Soft duration target. Used together with punctuation
            heuristics to pick natural break points. Default: 5.0 seconds.
        max_chars: Hard character cap on the full merged text (summed across
            lines). Default: 84 (2 × :data:`MAX_CHARS`). Controls when a new
            segment would overflow the group and force a flush.
        line_max_chars: Per-line wrap width passed to :func:`wrap_text`.
            Default: :data:`MAX_CHARS` (42) — the YouTube 16:9 convention.
            Independent of ``max_chars`` because the budget is a total across
            up to 2 lines while the wrap is per-line.

    Returns:
        A new list of subtitles representing the merged groups. An empty input
        yields an empty list. A single-subtitle input yields the same subtitle
        normalized through ``wrap_text``.
    """
    if not subtitles:
        return []

    result: list[srt.Subtitle] = []
    group: list[srt.Subtitle] = []
    target_delta = datetime.timedelta(seconds=target_duration_s)
    pending_overflow = ""  # text that didn't fit in the previous flush — re-attached to the next sub

    for sub in subtitles:
        # Re-attach any overflow from the previous flush to the start of this sub.
        sub = _prepend_overflow(sub, pending_overflow)
        pending_overflow = ""

        if not group:
            group.append(sub)
            continue

        candidate_text_len = len(_joined_text(group + [sub]))
        if candidate_text_len > max_chars:
            flushed, pending_overflow = _flush(group, line_max_chars)
            result.append(flushed)
            # The overflow belongs to the *previous* group; carry it onto the
            # current sub before opening the new group.
            sub = _prepend_overflow(sub, pending_overflow)
            pending_overflow = ""
            group = [sub]
            continue

        group.append(sub)

        current_duration = group[-1].end - group[0].start
        if current_duration >= target_delta and _ends_with_strong_punct(group[-1].content):
            flushed, pending_overflow = _flush(group, line_max_chars)
            result.append(flushed)
            group = []

    if group:
        flushed, pending_overflow = _flush(group, line_max_chars)
        result.append(flushed)

    # End-of-video tail: nowhere to push the overflow. Try to re-fit it into
    # the last segment (may produce a 3-line content, accepted as the lesser
    # evil vs total content loss). Log a warning so the operator can review.
    if pending_overflow and result:
        last = result[-1]
        existing = " ".join(last.content.split())  # flatten current 1-2 line content
        combined = (existing + " " + pending_overflow).strip()
        new_content, residual = wrap_text_with_overflow(combined, max_chars=line_max_chars)
        if residual:
            logger.warning(
                "merge_into_sentences: end-of-video tail did not fit even after "
                "re-attaching to the last segment. Lost: %r",
                residual,
            )
        result[-1] = srt.Subtitle(
            index=last.index,
            start=last.start,
            end=last.end,
            content=new_content,
            proprietary=last.proprietary,
        )

    return result
