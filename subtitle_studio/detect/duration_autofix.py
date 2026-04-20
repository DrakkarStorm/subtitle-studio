"""Auto-merge of too-short segments (duration < error threshold)."""

from __future__ import annotations

import srt

from ..generate.subtitle import MAX_CHARS, wrap_text
from .guidelines import DEFAULT_MIN_DURATION_ERROR_S
from .models import DurationAutoFix


def auto_merge_short_segments(
    subtitles: list[srt.Subtitle],
    min_duration_s: float = DEFAULT_MIN_DURATION_ERROR_S,
    max_chars: int = MAX_CHARS,
) -> tuple[list[srt.Subtitle], list[DurationAutoFix]]:
    """Merge segments whose duration is below min_duration_s.

    - Preferred direction: merge into the previous segment (backward).
    - If the short segment is the first one (no previous): merge into the next
      one (forward) as soon as a normal segment is encountered.
    - The merged content is re-wrapped with wrap_text(max_chars).
    - Indices are not renumbered — write_srt handles renumbering via
      srt.compose(reindex=True).

    Pathological case — **all segments are short**: no merge has a normal
    anchor to apply. Those segments are then returned as-is (via
    ``pending_forward``), with no merge and no entry in ``fixes``. The caller
    then sees an unchanged SRT and uncorrected duration violations — this
    case is rare (requires an entire SRT below threshold) but is not covered
    by the fix and must be handled upstream if needed.

    Does not mutate the original objects.
    """
    sorted_subs = sorted(subtitles, key=lambda s: s.start)
    result: list[srt.Subtitle] = []
    fixes: list[DurationAutoFix] = []
    pending_forward: list[srt.Subtitle] = []  # segments waiting for a forward merge

    for sub in sorted_subs:
        duration_s = (sub.end - sub.start).total_seconds()

        if duration_s >= min_duration_s:
            # Normal segment — absorb any pending (forward merge)
            if pending_forward:
                all_parts = pending_forward + [sub]
                merged_text = wrap_text(
                    " ".join(s.content.replace("\n", " ") for s in all_parts),
                    max_chars,
                )
                merged = srt.Subtitle(
                    index=sub.index,
                    start=pending_forward[0].start,
                    end=sub.end,
                    content=merged_text,
                    proprietary=sub.proprietary,
                )
                result.append(merged)
                for s in pending_forward:
                    fixes.append(
                        DurationAutoFix(
                            segment=s.index,
                            merged_with=sub.index,
                            original_duration_s=round((s.end - s.start).total_seconds(), 2),
                        )
                    )
                pending_forward.clear()
            else:
                result.append(sub)
        else:
            # Short segment
            if result:
                # Backward merge
                base = result.pop()
                merged_text = wrap_text(
                    base.content.replace("\n", " ") + " " + sub.content.replace("\n", " "),
                    max_chars,
                )
                merged = srt.Subtitle(
                    index=base.index,
                    start=base.start,
                    end=sub.end,
                    content=merged_text,
                    proprietary=base.proprietary,
                )
                result.append(merged)
                fixes.append(
                    DurationAutoFix(
                        segment=sub.index,
                        merged_with=base.index,
                        original_duration_s=round(duration_s, 2),
                    )
                )
            else:
                # No previous — defer the forward merge
                pending_forward.append(sub)

    # Pathological case: all segments are short, none could be merged
    result.extend(pending_forward)

    return result, fixes
