"""SRT/VTT formatters with YouTube-compliant line wrapping."""

import textwrap
from datetime import timedelta
from typing import Any, cast

import srt

MAX_CHARS = 42  # standard YouTube convention (16:9)
MAX_LINES = 2  # YouTube silently truncates anything beyond 2 lines


def wrap_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Wrap subtitle text according to YouTube conventions.

    - Wraps to `max_chars` characters per line (default 42, YouTube 16:9 standard;
      use 32 for 9:16 Shorts).
    - Truncates to `MAX_LINES` lines (2). Any extra content is **lost** — the
      caller is responsible for ensuring the text fits (via CPS splitting or
      semantic shortening).
    - Never breaks a word (`break_long_words=False`): a word longer than
      `max_chars` silently overflows.

    Args:
        text: Raw text to wrap (internal `\\n` are preserved by `wrap`).
        max_chars: Maximum line width. Must be ≥ 10.

    Raises:
        ValueError: If `max_chars < 10`.
    """
    if max_chars < 10:
        raise ValueError(f"max_chars must be ≥ 10, got: {max_chars}")
    lines = textwrap.wrap(text.strip(), width=max_chars, break_long_words=False)
    return "\n".join(lines[:MAX_LINES])


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
