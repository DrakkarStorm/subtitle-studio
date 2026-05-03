"""SRT/VTT formatters with YouTube-compliant line wrapping."""

import logging
import textwrap
from datetime import timedelta
from typing import Any, cast

import srt

logger = logging.getLogger(__name__)

MAX_CHARS = 50  # extended convention — covers French sentences with long words
# without forcing 3-line truncation. Standard YouTube 16:9 is 42;
# we raise to 50 to absorb word-boundary overflow that 42 cannot
# handle in 2 lines (see /tmp/merger_demo.py and the wrap_text
# truncation log analysis).
MAX_LINES = 2  # YouTube silently truncates anything beyond 2 lines


def wrap_text_with_overflow(text: str, max_chars: int = MAX_CHARS) -> tuple[str, str]:
    """Wrap text into ≤ MAX_LINES lines and return any leftover content.

    Unlike :func:`wrap_text`, this function does **not** drop content when the
    input wraps to more than ``MAX_LINES`` lines. Instead, the lines that don't
    fit are returned as a single space-joined ``overflow`` string so callers
    can re-attach them (e.g. to the next subtitle segment).

    Returns:
        ``(content, overflow)`` where ``content`` is the kept text joined by
        ``\\n`` (≤ ``MAX_LINES`` lines, each ≤ ``max_chars``) and ``overflow``
        is the remainder (``""`` when nothing overflowed).

    Raises:
        ValueError: If ``max_chars < 10``.
    """
    if max_chars < 10:
        raise ValueError(f"max_chars must be ≥ 10, got: {max_chars}")
    lines = textwrap.wrap(text.strip(), width=max_chars, break_long_words=False)
    content = "\n".join(lines[:MAX_LINES])
    overflow = " ".join(lines[MAX_LINES:])
    return content, overflow


def wrap_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Wrap subtitle text according to YouTube conventions.

    - Wraps to `max_chars` characters per line (default 42, YouTube 16:9 standard;
      use 32 for 9:16 Shorts).
    - Truncates to `MAX_LINES` lines (2). Any extra content is **lost** — the
      caller is responsible for ensuring the text fits (via CPS splitting or
      semantic shortening). Use :func:`wrap_text_with_overflow` if you want to
      preserve and re-attach the overflow.
    - Never breaks a word (`break_long_words=False`): a word longer than
      `max_chars` silently overflows.

    Args:
        text: Raw text to wrap (internal `\\n` are preserved by `wrap`).
        max_chars: Maximum line width. Must be ≥ 10.

    Raises:
        ValueError: If `max_chars < 10`.
    """
    content, overflow = wrap_text_with_overflow(text, max_chars=max_chars)
    if overflow:
        logger.warning(
            "wrap_text dropped %d char(s) from a %d-char input [max_chars=%d]: kept=%r dropped=%r",
            len(overflow),
            len(text.strip()),
            max_chars,
            content,
            overflow,
        )
    return content


def to_subtitles(result: Any, max_chars: int = MAX_CHARS) -> list[srt.Subtitle]:
    """Convert a stable-ts result into a list of ``srt.Subtitle`` objects.

    Exposes the intermediate list so callers can apply post-processing steps
    (sentence merging, auto-fixes, etc.) before composing the final SRT.

    Line wrapping is applied here so the returned subtitles already respect
    the YouTube convention (``max_chars`` per line, ``MAX_LINES`` lines max).
    """
    return [
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=seg.start),
            end=timedelta(seconds=seg.end),
            content=wrap_text(seg.text, max_chars),
        )
        for i, seg in enumerate(result.segments, start=1)
    ]


def to_srt(result: Any, max_chars: int = MAX_CHARS) -> str:
    """Convert a stable-ts result to SRT string with YouTube-compliant line wrapping."""
    return cast(str, srt.compose(to_subtitles(result, max_chars)))


def _vtt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def to_vtt(result: Any, max_chars: int = MAX_CHARS) -> str:
    """Convert a stable-ts result to WebVTT string with YouTube-compliant line wrapping."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(result.segments, start=1):
        lines += [
            str(i),
            f"{_vtt_time(seg.start)} --> {_vtt_time(seg.end)}",
            wrap_text(seg.text, max_chars),
            "",
        ]
    return "\n".join(lines)
