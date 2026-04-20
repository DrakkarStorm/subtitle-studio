"""SRT file read, write and correction helpers."""

from __future__ import annotations

from pathlib import Path

import srt

from .models import Correction, SRTParseError

# Encodings tried in order (utf-8-sig handles the Windows BOM)
_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


def parse_srt(path: Path) -> list[srt.Subtitle]:
    """Read an SRT file trying multiple encodings.

    Encodings are tried in order until one yields a non-empty subtitle list:

    1. ``utf-8-sig`` — UTF-8 with BOM (commonly produced by Windows Notepad).
    2. ``utf-8`` — nominal case (pipeline output, modern exports).
    3. ``latin-1`` — legacy European SRTs without a BOM.
    4. ``cp1252`` — Windows legacy SRTs.

    A valid but **empty** file (0 subtitles) is treated as a parsing failure
    and falls through to the next encoding — then raises ``SRTParseError`` if
    all encodings fail.

    Returns:
        Ordered list of subtitles (file order, no sorting).

    Raises:
        SRTParseError: If the file does not exist, or if no encoding yields a
            valid non-empty parse.
    """
    if not path.exists():
        raise SRTParseError(f"File not found: '{path.name}'")

    for encoding in _ENCODINGS:
        try:
            content = path.read_text(encoding=encoding)
            subtitles = list(srt.parse(content))
            if subtitles:
                return subtitles
        except (UnicodeDecodeError, srt.SRTParseError):
            continue

    raise SRTParseError(f"Could not read '{path.name}'. Ensure the file is a valid SRT (utf-8, latin-1 or cp1252).")


def write_srt(subtitles: list[srt.Subtitle], path: Path) -> None:
    """Write subtitles to an SRT file.

    Uses an atomic write (temporary file + rename) to avoid producing a
    corrupted file if interrupted.
    """
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(srt.compose(subtitles), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def apply_corrections(
    subtitles: list[srt.Subtitle],
    corrections: list[Correction],
) -> list[srt.Subtitle]:
    """Apply corrections to the subtitle list.

    Does not mutate the originals — returns a new list.
    """
    patch: dict[int, str] = {c.segment: c.suggestion for c in corrections}

    result: list[srt.Subtitle] = []
    for sub in subtitles:
        if sub.index in patch:
            corrected = srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=patch[sub.index],
                proprietary=sub.proprietary,
            )
            result.append(corrected)
        else:
            result.append(sub)

    return result
