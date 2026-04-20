"""Tests for auto-merging of too-short segments."""

from __future__ import annotations

import datetime

import pytest
import srt

from subtitle_studio.detect.cps_autofix import auto_fix_cps_violations
from subtitle_studio.detect.duration_autofix import auto_merge_short_segments
from subtitle_studio.detect.guidelines import DEFAULT_MIN_DURATION_ERROR_S
from subtitle_studio.detect.models import DurationAutoFix


def _sub(
    index: int,
    start_s: float,
    end_s: float,
    content: str,
) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=datetime.timedelta(seconds=start_s),
        end=datetime.timedelta(seconds=end_s),
        content=content,
    )


# ---------------------------------------------------------------------------
# No merge — segments unchanged
# ---------------------------------------------------------------------------


class TestNoMerge:
    def test_no_short_segments_unchanged(self) -> None:
        """All segments ≥ 0.5s → returned unchanged, no fix."""
        subs = [
            _sub(1, 0, 1, "Bonjour le monde."),
            _sub(2, 2, 4, "Comment ça va ?"),
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 2
        assert len(fixes) == 0
        assert result[0].content == "Bonjour le monde."
        assert result[1].content == "Comment ça va ?"

    def test_boundary_exactly_0_5s_not_merged(self) -> None:
        """Duration exactly 0.5s → no merge (threshold is strict < 0.5s, R7)."""
        subs = [
            _sub(1, 0, 2, "Normal."),
            _sub(2, 3, 3.5, "Exact."),  # duration = 0.5s exactly
        ]
        result, fixes = auto_merge_short_segments(subs, min_duration_s=DEFAULT_MIN_DURATION_ERROR_S)
        assert len(result) == 2
        assert len(fixes) == 0

    def test_very_short_duration_zero(self) -> None:
        """Zero-duration segment → merged without crashing (degenerate case)."""
        subs = [
            _sub(1, 0, 2, "Normal."),
            _sub(2, 3, 3, "Zero."),  # duration = 0s
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 1
        assert len(fixes) == 1


# ---------------------------------------------------------------------------
# Backward merge (into the previous)
# ---------------------------------------------------------------------------


class TestBackwardMerge:
    def test_backward_merge_basic(self) -> None:
        """Short segment merged backward: timecodes extended correctly."""
        subs = [
            _sub(1, 0, 2, "Normal."),
            _sub(2, 2, 2.3, "Short."),  # 0.3s < 0.5s
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 1
        assert len(fixes) == 1
        merged = result[0]
        # Timecodes: start of base, end of the short segment
        assert merged.start == datetime.timedelta(seconds=0)
        assert merged.end == datetime.timedelta(seconds=2.3)

    def test_backward_merge_text_combined(self) -> None:
        """Merged text = base text + short segment text, wrap_text applied."""
        subs = [
            _sub(1, 0, 2, "Bonjour"),
            _sub(2, 2, 2.3, "monde."),
        ]
        result, fixes = auto_merge_short_segments(subs)
        merged = result[0]
        # The content must contain both texts
        content_flat = merged.content.replace("\n", " ")
        assert "Bonjour" in content_flat
        assert "monde." in content_flat

    def test_backward_merge_preserves_base_index(self) -> None:
        """Merged segment keeps the base's index (not the short segment's)."""
        subs = [
            _sub(1, 0, 2, "Base."),
            _sub(5, 2, 2.3, "Short."),  # intentionally different index
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert result[0].index == 1  # base's index, not 5

    def test_backward_merge_fix_recorded(self) -> None:
        """DurationAutoFix recorded with segment=short.index, direction='backward'."""
        subs = [
            _sub(1, 0, 2, "Base."),
            _sub(2, 2, 2.3, "Short."),
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(fixes) == 1
        fix = fixes[0]
        assert isinstance(fix, DurationAutoFix)
        assert fix.segment == 2
        assert fix.merged_with == 1
        assert fix.original_duration_s == pytest.approx(0.3, abs=0.001)
        # Direction derivable: fix.segment > fix.merged_with → backward

    def test_backward_merge_long_text_truncated_to_two_lines(self) -> None:
        """Merging two long segments → wrap_text caps at 2 lines max (silent truncation).

        Known behavior: content beyond 2 lines is silently truncated by wrap_text
        (lines[:MAX_LINES]). This test documents that behavior to keep it visible.
        """
        subs = [
            _sub(1, 0, 3, "A" * 50),  # 50 chars valid
            _sub(2, 3, 3.3, "B" * 50),  # 50 chars — 0.3s → backward merge
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 1
        assert len(fixes) == 1
        lines = result[0].content.split("\n")
        assert len(lines) <= 2  # wrap_text caps at MAX_LINES=2 — no crash

    def test_last_segment_short_merges_backward(self) -> None:
        """Last segment (no successor) short → backward merge."""
        subs = [
            _sub(1, 0, 3, "Intro."),
            _sub(2, 4, 6, "Body."),
            _sub(3, 7, 7.4, "End."),  # 0.4s < 0.5s
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 2
        assert len(fixes) == 1
        assert fixes[0].segment == 3
        assert fixes[0].merged_with == 2


# ---------------------------------------------------------------------------
# Forward merge (into the next)
# ---------------------------------------------------------------------------


class TestForwardMerge:
    def test_forward_merge_first_segment(self) -> None:
        """First segment short → forward merge: surviving index and timecodes."""
        subs = [
            _sub(1, 0, 0.3, "Short."),  # 0.3s < 0.5s, first
            _sub(2, 1, 3, "Normal."),
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 1
        merged = result[0]
        # The surviving segment takes the base's index (the normal segment)
        assert merged.index == 2
        # Timecodes: start of the short, end of the normal
        assert merged.start == datetime.timedelta(seconds=0)
        assert merged.end == datetime.timedelta(seconds=3)

    def test_forward_merge_fix_recorded(self) -> None:
        """DurationAutoFix recorded with direction='forward'."""
        subs = [
            _sub(1, 0, 0.3, "Short."),
            _sub(2, 1, 3, "Normal."),
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(fixes) == 1
        fix = fixes[0]
        assert fix.segment == 1
        assert fix.merged_with == 2
        assert fix.original_duration_s == pytest.approx(0.3, abs=0.001)
        # Direction derivable: fix.segment < fix.merged_with → forward

    def test_multiple_leading_short_segments(self) -> None:
        """Two leading short segments → both merged forward into the third."""
        subs = [
            _sub(1, 0, 0.2, "One."),
            _sub(2, 0.3, 0.45, "Two."),
            _sub(3, 1, 3, "Normal."),
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 1
        assert len(fixes) == 2
        assert all(f.merged_with == 3 for f in fixes)
        # Direction derivable: all fixes have segment < merged_with → forward
        assert result[0].index == 3


# ---------------------------------------------------------------------------
# Chained merges
# ---------------------------------------------------------------------------


class TestChainedMerge:
    def test_chained_short_segments_backward(self) -> None:
        """Two consecutive short segments → both merged backward in one pass.

        Scenario: [Normal(1), Short(2, 0.3s), Short(3, 0.4s)]
        Short(2) merges backward into Normal(1) → partial result [Merged(1)]
        Short(3) merges backward into Merged(1) → final result [Merged(1)]
        """
        subs = [
            _sub(1, 0, 2, "Normal."),
            _sub(2, 2, 2.3, "Short2."),  # 0.3s
            _sub(3, 2.3, 2.7, "Short3."),  # 0.4s
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 1
        assert len(fixes) == 2
        assert result[0].index == 1
        # Timecodes cover the whole span
        assert result[0].start == datetime.timedelta(seconds=0)
        assert result[0].end == datetime.timedelta(seconds=2.7)

    def test_all_segments_short_no_crash(self) -> None:
        """All segments short → no crash, returned as-is."""
        subs = [
            _sub(1, 0, 0.3, "One."),
            _sub(2, 0.4, 0.7, "Two."),
        ]
        result, fixes = auto_merge_short_segments(subs)
        # Both are in pending_forward, never merged
        assert len(result) == 2
        assert len(fixes) == 0


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------


class TestUnsortedInput:
    def test_unsorted_input_sorted_by_start(self) -> None:
        """Unsorted input → sorted by start before processing (no crash, correct result)."""
        subs = [
            _sub(3, 10, 12, "Third."),
            _sub(1, 0, 2, "First."),  # index=1 but positioned second
            _sub(2, 2, 2.3, "Short."),  # 0.3s < 0.5s — must merge with First
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert len(result) == 2
        assert len(fixes) == 1
        # The merged segment keeps the base's index (idx=1, "First")
        assert result[0].index == 1


class TestIndexPreservation:
    def test_no_renumbering_in_function(self) -> None:
        """auto_merge_short_segments does not renumber indices — key pipeline principle."""
        subs = [
            _sub(10, 0, 2, "Base."),  # intentionally non-sequential index
            _sub(20, 2, 2.3, "Short."),
        ]
        result, fixes = auto_merge_short_segments(subs)
        # The surviving index is the base's (10), not 1
        assert result[0].index == 10
        assert len(result) == 1

    def test_duration_after_merge_before_cps(self) -> None:
        """Merged segment with CPS > 21 → auto_fix_cps_violations handles it correctly.

        Verifies the merge → CPS fix interaction (correct pipeline order).
        A short segment (0.3s, "OK") merged backward into a long segment (2s, 40 chars)
        yields a 2.3s segment with ~18 chars — resulting CPS < 21, no CPS fix.
        """
        subs = [
            _sub(1, 0, 2, "A" * 40),  # 40 chars over 2s = 20 CPS — borderline ok
            _sub(2, 2, 2.3, "End."),  # 0.3s < 0.5s → merged
        ]
        merged_result, duration_fixes = auto_merge_short_segments(subs)
        assert len(duration_fixes) == 1

        # The merged segment goes through CPS auto-fix without triggering a fix
        cps_result, cps_fixes = auto_fix_cps_violations(merged_result)
        assert len(cps_fixes) == 0  # merged CPS is acceptable


# ---------------------------------------------------------------------------
# DurationAutoFix model
# ---------------------------------------------------------------------------


class TestReportModel:
    def test_duration_autofix_fields(self) -> None:
        """DurationAutoFix has the 3 required fields."""
        fix = DurationAutoFix(
            segment=3,
            merged_with=2,
            original_duration_s=0.3,
        )
        assert fix.segment == 3
        assert fix.merged_with == 2
        assert fix.original_duration_s == 0.3

    def test_original_duration_recorded(self) -> None:
        """fix.original_duration_s matches the segment's duration before merging."""
        subs = [
            _sub(1, 0, 2, "Base."),
            _sub(2, 2, 2.26, "Short."),  # 0.26s
        ]
        result, fixes = auto_merge_short_segments(subs)
        assert fixes[0].original_duration_s == pytest.approx(0.26, abs=0.001)
