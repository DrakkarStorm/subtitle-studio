"""Tests for sentence-level merging (YouTube-like segmentation)."""

from __future__ import annotations

import datetime

import srt

from subtitle_studio.generate.sentence_merger import (
    DEFAULT_MAX_CHARS,
    DEFAULT_TARGET_DURATION_S,
    merge_into_sentences,
)
from subtitle_studio.generate.subtitle import MAX_CHARS


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


def _duration_s(sub: srt.Subtitle) -> float:
    return (sub.end - sub.start).total_seconds()


# ---------------------------------------------------------------------------
# Happy path — merging
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_three_short_segments_below_target_merge_into_one(self) -> None:
        """3 segments totalling 4.5 s / ~60 chars, no punctuation → one merged group."""
        subs = [
            _sub(1, 0.0, 1.5, "Bonjour tout le monde"),
            _sub(2, 1.5, 3.0, "je voulais vous parler"),
            _sub(3, 3.0, 4.5, "de quelque chose"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 1
        merged = result[0]
        assert merged.start == datetime.timedelta(seconds=0.0)
        assert merged.end == datetime.timedelta(seconds=4.5)
        assert merged.index == 1
        flat = merged.content.replace("\n", " ")
        assert "Bonjour tout le monde" in flat
        assert "de quelque chose" in flat

    def test_splits_on_strong_punctuation_at_target(self) -> None:
        """Segment ending with '.' past the target duration closes the group."""
        subs = [
            _sub(1, 0.0, 2.5, "C'est vraiment génial"),
            _sub(2, 2.5, 5.2, "je vous le dis."),  # period at 5.2 s, above target
            _sub(3, 5.2, 7.0, "Voici la suite"),
        ]
        result = merge_into_sentences(subs, target_duration_s=DEFAULT_TARGET_DURATION_S)
        assert len(result) == 2
        assert result[0].end == datetime.timedelta(seconds=5.2)
        assert result[1].start == datetime.timedelta(seconds=5.2)

    def test_splits_on_strong_punctuation_with_exclamation(self) -> None:
        subs = [
            _sub(1, 0.0, 3.0, "Regardez ça"),
            _sub(2, 3.0, 5.5, "c'est incroyable!"),
            _sub(3, 5.5, 7.0, "On continue"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 2
        assert result[0].content.replace("\n", " ").endswith("c'est incroyable!")

    def test_splits_on_strong_punctuation_with_question_mark(self) -> None:
        subs = [
            _sub(1, 0.0, 3.0, "Tu sais quoi"),
            _sub(2, 3.0, 5.5, "ça te plaît?"),
            _sub(3, 5.5, 7.0, "Tant mieux"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input_returns_empty_list(self) -> None:
        assert merge_into_sentences([]) == []

    def test_single_subtitle_returned_wrapped(self) -> None:
        subs = [_sub(1, 0.0, 2.0, "Un petit texte")]
        result = merge_into_sentences(subs)
        assert len(result) == 1
        assert result[0].start == datetime.timedelta(seconds=0.0)
        assert result[0].end == datetime.timedelta(seconds=2.0)
        assert result[0].content == "Un petit texte"

    def test_first_segment_already_exceeds_max_chars_is_emitted_alone(self) -> None:
        """A single over-long segment is kept as its own group (merger does not split)."""
        long_text = "x " * 60  # ~120 chars before wrap
        subs = [
            _sub(1, 0.0, 2.0, long_text),
            _sub(2, 2.0, 3.0, "suite"),
        ]
        result = merge_into_sentences(subs)
        # The over-long first segment is flushed alone, the short one becomes its own group.
        assert len(result) == 2
        assert result[0].index == 1
        assert result[1].index == 2

    def test_no_strong_punctuation_still_flushes_at_end(self) -> None:
        """Input without any '.', '!', '?' still yields a result (end-of-iteration flush)."""
        subs = [
            _sub(1, 0.0, 2.0, "Sans ponctuation"),
            _sub(2, 2.0, 4.0, "on continue"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 1
        flat = result[0].content.replace("\n", " ")
        assert "Sans ponctuation" in flat
        assert "on continue" in flat

    def test_last_segment_ends_mid_phrase_still_flushed(self) -> None:
        """The trailing group is always flushed, even without a punctuation trigger."""
        subs = [
            _sub(1, 0.0, 4.0, "Début de phrase"),
            _sub(2, 4.0, 8.0, "sans fin"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) >= 1
        assert result[-1].end == datetime.timedelta(seconds=8.0)

    def test_punctuation_but_below_target_keeps_accumulating(self) -> None:
        """Strong punctuation below target duration does NOT trigger a flush."""
        # Duration below target (5 s) — even with a period, the group continues.
        subs = [
            _sub(1, 0.0, 1.5, "Court."),
            _sub(2, 1.5, 3.0, "encore court"),
            _sub(3, 3.0, 4.5, "toujours en dessous"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_cumulative_length_exactly_max_chars_flushes_before_overflow(self) -> None:
        """A candidate addition that would overflow max_chars triggers a flush.

        Sizes are chosen relative to DEFAULT_MAX_CHARS (= MAX_CHARS × 2 = 100):
        50 + 1 (space) + 49 = 100 chars fits at the boundary; adding any
        non-empty third sub pushes past 100 → flush before adding.
        """
        subs = [
            _sub(1, 0.0, 1.0, "a" * 50),
            _sub(2, 1.0, 2.0, "b" * 49),  # 50 + 1 + 49 = 100, ≤ 100
            _sub(3, 2.0, 3.0, "c" * 10),  # would push to 111 > 100 → flush before adding
        ]
        result = merge_into_sentences(subs)
        # First two merge (100 chars), third opens a new group.
        assert len(result) == 2
        assert result[1].index == 3

    def test_respects_custom_max_chars(self) -> None:
        """Caller can tighten the character budget."""
        subs = [
            _sub(1, 0.0, 1.0, "a" * 30),
            _sub(2, 1.0, 2.0, "b" * 30),  # would be 61 chars, exceeds max_chars=50
        ]
        result = merge_into_sentences(subs, max_chars=50)
        assert len(result) == 2

    def test_respects_custom_target_duration(self) -> None:
        """Caller can tighten the duration target to force earlier splits at punctuation."""
        subs = [
            _sub(1, 0.0, 1.5, "Un début."),
            _sub(2, 1.5, 3.0, "Ensuite."),  # 1.5 s ≥ target=1.0, period triggers flush
            _sub(3, 3.0, 4.5, "Fin"),
        ]
        result = merge_into_sentences(subs, target_duration_s=1.0)
        # Group 1: sub1 alone (since we enter the loop with empty group; when sub2 is added,
        # duration is 3.0 s ≥ 1.0 and ends with '.' → flush sub1+sub2 together).
        # Group 2: sub3 alone at end.
        assert len(result) == 2

    def test_defaults_match_published_constants(self) -> None:
        """Sanity: defaults aren't silently changed."""
        assert DEFAULT_TARGET_DURATION_S == 5.0
        assert DEFAULT_MAX_CHARS == MAX_CHARS * 2 == 100

    def test_line_max_chars_controls_wrap_width(self) -> None:
        """Custom line_max_chars is honoured by the internal wrap_text call.

        Regression test for the bug where _flush hardcoded MAX_CHARS and
        silently ignored the caller's intent.
        """
        subs = [
            _sub(1, 0.0, 1.0, "un deux trois"),
            _sub(2, 1.0, 2.0, "quatre cinq six"),
            _sub(3, 2.0, 3.0, "sept huit neuf"),
        ]
        result = merge_into_sentences(subs, max_chars=80, line_max_chars=20)
        assert len(result) == 1
        for line in result[0].content.splitlines():
            assert len(line) <= 20, f"Line too long with line_max_chars=20: {line!r}"

    def test_line_max_chars_default_is_max_chars(self) -> None:
        """Omitting line_max_chars uses the MAX_CHARS default."""
        # Text long enough to wrap at MAX_CHARS but not at larger widths.
        subs = [_sub(1, 0.0, 3.0, "word " * 18)]  # 90 chars
        result = merge_into_sentences(subs)
        assert len(result) == 1
        for line in result[0].content.splitlines():
            assert len(line) <= MAX_CHARS


# ---------------------------------------------------------------------------
# Integration with wrap_text / index / timing
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_merged_content_is_rewrapped(self) -> None:
        """Merged text goes through wrap_text(MAX_CHARS) so each line ≤ 42 chars."""
        subs = [
            _sub(1, 0.0, 2.0, "Phrase un deux trois"),
            _sub(2, 2.0, 4.0, "quatre cinq six"),
            _sub(3, 4.0, 6.0, "sept huit."),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 1
        for line in result[0].content.split("\n"):
            assert len(line) <= MAX_CHARS, f"Line too long: {line!r}"

    def test_merged_timecodes_span_first_start_to_last_end(self) -> None:
        subs = [
            _sub(1, 0.3, 1.8, "ab"),
            _sub(2, 1.8, 3.4, "cd"),
            _sub(3, 3.4, 5.0, "ef"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 1
        assert result[0].start == datetime.timedelta(seconds=0.3)
        assert result[0].end == datetime.timedelta(seconds=5.0)

    def test_first_segment_index_preserved(self) -> None:
        """Each merged group keeps the index of its first segment — no renumbering."""
        subs = [
            _sub(7, 0.0, 3.0, "Bonjour"),
            _sub(8, 3.0, 5.5, "le monde."),  # period + duration ≥ 5 s → flush
            _sub(9, 5.5, 7.0, "Suite"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 2
        assert result[0].index == 7  # first group starts at sub 7
        assert result[1].index == 9  # second group starts at sub 9

    def test_internal_newlines_are_flattened(self) -> None:
        """Whisper sometimes emits content with embedded newlines; merger must normalize."""
        subs = [
            _sub(1, 0.0, 2.0, "ligne un\nligne deux"),
            _sub(2, 2.0, 4.0, "suite"),
        ]
        result = merge_into_sentences(subs)
        assert len(result) == 1
        flat = result[0].content.replace("\n", " ")
        assert "ligne un" in flat
        assert "ligne deux" in flat
        assert "suite" in flat

    def test_does_not_mutate_input(self) -> None:
        subs = [
            _sub(1, 0.0, 2.0, "Un"),
            _sub(2, 2.0, 4.0, "deux."),
        ]
        subs_copy = [_sub(s.index, s.start.total_seconds(), s.end.total_seconds(), s.content) for s in subs]
        merge_into_sentences(subs)
        assert [s.content for s in subs] == [s.content for s in subs_copy]
        assert [s.start for s in subs] == [s.start for s in subs_copy]


# ---------------------------------------------------------------------------
# Realistic scenario — approximates observed YouTube compression ratio
# ---------------------------------------------------------------------------


class TestRealisticCompression:
    def test_many_short_whisper_segments_compress_significantly(self) -> None:
        """Given ~10 short Whisper-like segments, expect ≤ 4 merged groups."""
        subs = []
        t = 0.0
        # 10 segments, average ~1.2 s each, one period mid-way
        fragments = [
            "Je commence à parler",
            "un peu de contexte",
            "voilà la première idée.",
            "Ensuite je continue",
            "avec une autre pensée",
            "qui se développe",
            "pas mal.",
            "Dernière partie",
            "pour conclure",
            "voilà merci.",
        ]
        for i, text in enumerate(fragments, start=1):
            subs.append(_sub(i, t, t + 1.2, text))
            t += 1.2

        result = merge_into_sentences(subs)
        assert 2 <= len(result) <= 4
        # Each merged group respects the char budget.
        for merged in result:
            flat = merged.content.replace("\n", " ")
            assert len(flat) <= DEFAULT_MAX_CHARS
        # Total coverage is preserved.
        assert result[0].start == subs[0].start
        assert result[-1].end == subs[-1].end


# ---------------------------------------------------------------------------
# Overflow chaining (regression: words at the line-3 boundary must survive)
# ---------------------------------------------------------------------------


class TestOverflowChaining:
    def test_overflow_word_appears_in_next_segment(self) -> None:
        """When a group's wrapped text exceeds 2 lines, the leftover words
        must be prepended to the next segment, never silently dropped."""
        # 4 Whisper subs that sum to 98 chars — wraps to 3 lines at width 50.
        # The trailing "qu'AWS." would be dropped by the old merger.
        subs = [
            _sub(1, 0.0, 1.0, "J'ai choisi GCP parce que j'aime bien"),
            _sub(2, 1.0, 2.0, "ce que propose Google,"),
            _sub(3, 2.0, 3.0, "et parce que c'est plus niche qu'AWS."),
            _sub(4, 3.0, 5.0, "Donc ça veut dire moins de monde."),
        ]
        result = merge_into_sentences(subs)

        # Total reconstructed text contains every word — no loss.
        flat = " ".join(s.content.replace("\n", " ") for s in result)
        assert "qu'AWS." in flat
        assert "Donc ça veut dire" in flat

        # The orphan word landed at the start of a later segment, not in the
        # truncated first one.
        first_flat = result[0].content.replace("\n", " ")
        assert "qu'AWS." not in first_flat

    def test_no_segment_exceeds_two_lines_with_overflow(self) -> None:
        """Even when chaining overflow, no segment should display > MAX_LINES lines."""
        subs = [
            _sub(1, 0.0, 1.0, "pas l'inverse."),
            _sub(2, 1.0, 4.0, "Simplement parce que j'apprends beaucoup mieux en pratiquant."),
            _sub(3, 4.0, 5.5, "Donc, j'ouvrais un lab."),
            _sub(4, 5.5, 7.0, "Et quand quelque chose."),
        ]
        result = merge_into_sentences(subs)
        for s in result:
            line_count = s.content.count("\n") + 1
            assert line_count <= 2, f"Segment exceeds 2 lines: {s.content!r}"

    def test_no_new_segments_added_by_overflow(self) -> None:
        """The chaining must redistribute content within the same segment count
        — pushing overflow forward, never spawning a new segment."""
        # Same input as case #1 above; group budget would close after sub 3
        # (98 chars > 100? actually 98 < 100, accepted as one group, then
        # punctuation triggers flush). The next sub starts a new group naturally.
        subs = [
            _sub(1, 0.0, 1.0, "J'ai choisi GCP parce que j'aime bien"),
            _sub(2, 1.0, 2.0, "ce que propose Google,"),
            _sub(3, 2.0, 6.0, "et parce que c'est plus niche qu'AWS."),
            _sub(4, 6.0, 8.0, "Donc ça veut dire moins de monde."),
        ]
        result = merge_into_sentences(subs)
        # Pre-fix, the old behavior produced 2 segments with "qu'AWS." dropped.
        # Post-fix, still 2 segments, but content is preserved via overflow.
        assert len(result) == 2

    def test_overflow_at_end_of_video_logs_residual(self, caplog) -> None:
        """If overflow remains after the very last segment, log a warning
        rather than silently lose content."""
        import logging

        # A single very long sub that wraps to 4 lines at width 50 — the
        # second flush has nothing to push to, the tail must surface.
        long_text = "word " * 50  # 250 chars → 5 lines @ width 50
        subs = [_sub(1, 0.0, 5.0, long_text.strip())]
        with caplog.at_level(logging.WARNING, logger="subtitle_studio.generate.sentence_merger"):
            result = merge_into_sentences(subs)
        # Single output segment, MAX_LINES respected for the saved content.
        assert len(result) == 1
        # The end-of-video residual triggers a warning.
        assert any("end-of-video" in rec.message.lower() for rec in caplog.records)
