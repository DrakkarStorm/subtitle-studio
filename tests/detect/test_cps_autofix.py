"""Tests for deterministic CPS auto-correction."""

from __future__ import annotations

import datetime

import srt

from subtitle_studio.detect.cps_autofix import auto_fix_cps_violations
from subtitle_studio.detect.guidelines import DEFAULT_MAX_CPS, DEFAULT_MIN_DURATION_ERROR_S
from subtitle_studio.detect.models import CpsAutoFix
from subtitle_studio.generate.subtitle import wrap_text


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


def _cps(sub: srt.Subtitle) -> float:
    duration = (sub.end - sub.start).total_seconds()
    return len(sub.content.replace("\n", " ")) / duration


# ---------------------------------------------------------------------------
# No violations — segments unchanged
# ---------------------------------------------------------------------------


class TestNoViolations:
    def test_no_violations_unchanged(self) -> None:
        """Valid segments are returned unchanged, with no fix."""
        subs = [
            _sub(1, 0, 3, "Bonjour le monde."),  # ~6 CPS — ok
            _sub(2, 4, 6, "Comment ça va ?"),  # ~7.5 CPS — ok
        ]
        result, fixes = auto_fix_cps_violations(subs)
        assert len(result) == 2
        assert len(fixes) == 0
        assert result[0].content == "Bonjour le monde."
        assert result[1].content == "Comment ça va ?"

    def test_warnings_not_fixed(self) -> None:
        """CPS warning segments (18–21 chars/s) are not modified."""
        # 38 chars over 2s = 19 CPS — warning, not error
        content = "A" * 38
        subs = [_sub(1, 0, 2, content)]
        result, fixes = auto_fix_cps_violations(subs, max_cps=DEFAULT_MAX_CPS)
        assert len(result) == 1
        assert result[0].content == content
        assert len(fixes) == 0

    def test_very_short_duration_skipped(self) -> None:
        """Segment < 0.2s (below the CPS evaluation threshold) is skipped even with high CPS."""
        subs = [_sub(1, 0, 0.1, "A" * 50)]
        result, fixes = auto_fix_cps_violations(subs)
        assert len(result) == 1
        assert len(fixes) == 0


# ---------------------------------------------------------------------------
# Successful split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_split_two_words_valid_gap(self) -> None:
        """2-word segment + enough gap → 2 sub-segments with CPS ≤ 21.

        Checked data:
          "Bonjour monde" = 13 chars over 0.5s = 26 CPS > 21
          gap = 4.5s → borrow = min(4.5×0.8, 1.0) = 1.0s → window = 1.5s
          feasibility: 13 / 1.5 = 8.7 ≤ 21 ✓
          split: "Bonjour"(7) / "monde"(5) → effective = 12
          t1 = 1.5×7/12 = 0.875s ≥ 0.5 ✓  t2 = 0.625s ≥ 0.5 ✓
        """
        subs = [
            _sub(1, 0, 0.5, "Bonjour monde"),
            _sub(2, 5, 7, "Suite du contenu."),
        ]
        result, fixes = auto_fix_cps_violations(subs)

        assert len(fixes) == 1
        assert fixes[0].action == "split"
        assert fixes[0].segment == 1

        split_subs = result[:-1]  # exclude the unchanged segment 2
        for s in split_subs:
            assert _cps(s) <= DEFAULT_MAX_CPS + 0.01

    def test_split_assigns_proportional_time(self) -> None:
        """Time allocated to each sub-segment is proportional to its chars.

        Checked data:
          "Bonjour monde" = 13 chars over 0.5s = 26 CPS > 21
          window = 1.5s (same calc as above)
          split: text1="Bonjour"(7), text2="monde"(5), effective=12
          t1 = 1.5×7/12 ≈ 0.875s   t2 = 1.5×5/12 ≈ 0.625s
          ratio t1/(t1+t2) = 7/12 ≈ 0.583
        """
        subs = [
            _sub(1, 0, 0.5, "Bonjour monde"),
            _sub(2, 5, 7, "Suite."),
        ]
        result, fixes = auto_fix_cps_violations(subs)
        assert fixes[0].action == "split"

        sub1 = result[0]
        sub2 = result[1]
        t1 = (sub1.end - sub1.start).total_seconds()
        t2 = (sub2.end - sub2.start).total_seconds()

        chars1 = len(sub1.content.replace("\n", " "))
        chars2 = len(sub2.content.replace("\n", " "))
        expected_ratio = chars1 / (chars1 + chars2)
        actual_ratio = t1 / (t1 + t2)
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_split_applies_wrap_text(self) -> None:
        """Each sub-segment content goes through wrap_text(max_chars).

        Checked data:
          "Hello world wonderful test" = 26 chars over 0.5s = 52 CPS > 21
          gap = 4.5s → borrow = 1.0s → window = 1.5s
          feasibility: 26 / 1.5 = 17.3 ≤ 21 ✓
          split at word boundary closest to the middle (13 chars):
            i=1: "Hello"(5) → dist 8
            i=2: "Hello world"(11) → dist 2  ← best
            i=3: "Hello world wonderful"(21) → dist 8
          text1="Hello world"(11), text2="wonderful test"(14), effective=25
          t1 = 1.5×11/25 = 0.66s ≥ 0.5 ✓   t2 = 1.5×14/25 = 0.84s ≥ 0.5 ✓
          With max_chars=10: wrap_text("Hello world", 10) = "Hello\\nworld"
        """
        subs = [
            _sub(1, 0, 0.5, "Hello world wonderful test"),
            _sub(2, 5, 7, "Fin."),
        ]
        result, fixes = auto_fix_cps_violations(subs, max_chars=10)

        assert fixes[0].action == "split"
        # Each sub-segment content must be wrapped
        for s in result[:-1]:
            raw = s.content.replace("\n", " ")
            assert s.content == wrap_text(raw, 10)

    def test_split_does_not_renumber_indices(self) -> None:
        """After multiple splits, original indices are preserved (no renumbering).

        Sequential renumbering is delegated to srt.compose(reindex=True) in write_srt.
        Two split segments (idx=1 and idx=2) + one normal segment (idx=3):
        expected result: [idx=1, idx=1, idx=2, idx=2, idx=3]
        """
        subs = [
            _sub(1, 0, 0.5, "Bonjour monde"),  # split → 2 sub-segments (idx=1)
            _sub(2, 5, 5.5, "Hello world"),  # split → 2 sub-segments (idx=2)
            _sub(3, 10, 12, "Normal."),  # unchanged (idx=3)
        ]
        result, fixes = auto_fix_cps_violations(subs)

        split_fixes = [f for f in fixes if f.action == "split"]
        assert len(split_fixes) == 2
        assert len(result) == 5

        # Original indices are preserved — no renumbering
        assert result[0].index == 1
        assert result[1].index == 1  # second sub-segment of split on idx=1
        assert result[2].index == 2
        assert result[3].index == 2  # second sub-segment of split on idx=2
        assert result[4].index == 3

    def test_cps_fixes_returned_correctly(self) -> None:
        """CpsAutoFix returned with the correct fields."""
        subs = [
            _sub(1, 0, 0.5, "Bonjour monde"),
            _sub(2, 5, 7, "Suite."),
        ]
        result, fixes = auto_fix_cps_violations(subs)

        assert len(fixes) >= 1
        fix = fixes[0]
        assert isinstance(fix, CpsAutoFix)
        assert fix.segment == 1
        assert fix.action in ("split", "downgrade")
        assert fix.original_cps > DEFAULT_MAX_CPS

    def test_last_segment_uses_two_second_extension(self) -> None:
        """Last segment (no successor) → +2s extension → can be split.

        "Bonjour monde coucou test" = 25 chars over 0.5s = 50 CPS > 21
        No successor → extension = 2s → window = 2.5s
        feasibility: 25 / 2.5 = 10 ≤ 21 ✓
        split: "Bonjour monde"(13) / "coucou test"(11) effective=24
        t1 = 2.5×13/24 ≈ 1.35s ≥ 0.5 ✓   t2 ≈ 1.15s ≥ 0.5 ✓
        """
        subs = [_sub(1, 0, 0.5, "Bonjour monde coucou test")]
        result, fixes = auto_fix_cps_violations(subs)

        assert len(fixes) == 1
        assert fixes[0].action == "split"
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Downgrade (irreducible)
# ---------------------------------------------------------------------------


class TestDowngrade:
    def test_downgrade_single_word(self) -> None:
        """1-word segment → action=downgrade, wrap_text applied (no-op: 19 chars < max_chars)."""
        # 19 chars over 0.5s = 38 CPS > 21; single word → irreducible
        subs = [_sub(1, 0, 0.5, "Supercalifragiliste")]
        result, fixes = auto_fix_cps_violations(subs)

        assert len(fixes) == 1
        assert fixes[0].action == "downgrade"
        assert result[0].content == "Supercalifragiliste"

    def test_downgrade_window_insufficient(self) -> None:
        """Window insufficient even with extension → downgrade.

        100 chars over 0.5s = 200 CPS; no successor → +2s → window = 2.5s
        feasibility: 100 / 2.5 = 40 > 21 → infeasible
        """
        content = "A" * 100
        subs = [_sub(1, 0, 0.5, content)]
        result, fixes = auto_fix_cps_violations(subs)

        assert fixes[0].action == "downgrade"
        assert result[0].content == content

    def test_downgrade_subsegment_too_short(self) -> None:
        """Split producing sub-segment < min_duration_s → downgrade.

        "Abc Defghi" = 10 chars over 0.4s = 25 CPS > 21
        gap = 0.5s → borrow = min(0.5×0.8, 1.0) = 0.4s → window = 0.8s
        feasibility: 10/0.8 = 12.5 ≤ 21 ✓ (technically feasible)
        split: "Abc"(3) / "Defghi"(6) effective=9
        t1 = 0.8×3/9 = 0.267s < 0.5 → too short → downgrade
        """
        subs = [
            _sub(1, 0.0, 0.4, "Abc Defghi"),
            _sub(2, 0.9, 2.0, "Suite."),
        ]
        result, fixes = auto_fix_cps_violations(subs, min_duration_s=DEFAULT_MIN_DURATION_ERROR_S)

        assert fixes[0].action == "downgrade"

    def test_downgrade_preserves_content(self) -> None:
        """Downgraded segment → wrap_text applied to content (no-op: 19 chars < max_chars=42)."""
        original_content = "Supercalifragiliste"
        subs = [_sub(1, 0, 0.5, original_content)]
        result, fixes = auto_fix_cps_violations(subs)

        assert fixes[0].action == "downgrade"
        assert result[0].content == original_content

    def test_downgrade_zero_gap(self) -> None:
        """Zero gap (contiguity) → borrow=0 → window = original duration → infeasible.

        "AAAA BBB" = 8 chars over 0.3s = 26.7 CPS > 21
        gap = 0s → borrow = 0 → total_window_s = 0.3s
        feasibility: 8/0.3 = 26.7 > 21 → infeasible → downgrade
        """
        subs = [
            _sub(1, 0.0, 0.3, "AAAA BBB"),
            _sub(2, 0.3, 1.0, "Suite."),  # gap = 0
        ]
        result, fixes = auto_fix_cps_violations(subs)

        assert fixes[0].action == "downgrade"

    def test_downgrade_applies_wrap_text(self) -> None:
        """Downgraded segment → wrap_text(max_chars) applied to content, timecodes preserved.

        Checked data:
          "C'est super utile pour ceux qui utilisent kubectl." = 50 chars over 0.5s = 100 CPS > 21
          gap = 0s → borrow = 0 → window = 0.5s → feasibility: 100 > 21 → forced downgrade
          With max_chars=32: wrap_text yields 2 lines (line 1 = 31 chars ≤ 32 ✓)
        """
        content = "C'est super utile pour ceux qui utilisent kubectl."
        subs = [
            _sub(1, 0.0, 0.5, content),
            _sub(2, 0.5, 2.0, "Suite."),  # gap = 0 → borrow = 0 → infeasible window
        ]
        result, fixes = auto_fix_cps_violations(subs, max_chars=32)

        assert fixes[0].action == "downgrade"
        assert result[0].content == wrap_text(content, 32)
        assert result[0].index == 1
        assert result[0].start == subs[0].start
        assert result[0].end == subs[0].end

    def test_downgrade_after_split_preserves_original_index(self) -> None:
        """A split on segment N must not alter the index of segment N+1 (downgrade).

        Regression P1-C: without renumbering in auto_fix_cps_violations,
        CpsAutoFix.segment for the downgrade = original index = what check_guidelines sees.
        If renumbering were done here, the downgrade would be recorded as segment 2
        but check_guidelines would see it renumbered to 3 → _apply_downgrades would miss it.

        Scenario:
          sub1 (idx=1) "Bonjour monde" → split → two subs idx=1
          sub2 (idx=2) "Supercalifragiliste" → downgrade → sub idx=2 unchanged
        Expected result: downgrade_fix.segment == 2, result[2].index == 2
        """
        subs = [
            _sub(1, 0, 0.5, "Bonjour monde"),  # split
            _sub(2, 5, 5.5, "Supercalifragiliste"),  # downgrade (single word)
        ]
        result, fixes = auto_fix_cps_violations(subs)

        assert len(fixes) == 2
        split_fix = next(f for f in fixes if f.action == "split")
        downgrade_fix = next(f for f in fixes if f.action == "downgrade")

        assert split_fix.segment == 1
        assert downgrade_fix.segment == 2  # must be 2, not 3 (no renumbering)

        # The downgraded segment in the result keeps its original index
        downgrade_sub = result[-1]  # last element (not split)
        assert downgrade_sub.index == 2
