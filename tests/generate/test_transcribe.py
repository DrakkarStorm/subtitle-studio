"""Unit tests for the transcription dispatcher and hallucination filter."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from subtitle_studio.generate.transcribe import drop_whisper_hallucinations


def _seg(start: float, end: float, text: str) -> SimpleNamespace:
    """Build a fake Whisper segment compatible with the filter API."""
    return SimpleNamespace(start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# drop_whisper_hallucinations
# ---------------------------------------------------------------------------


class TestDropWhisperHallucinations:
    def test_empty_input_returns_empty(self) -> None:
        assert drop_whisper_hallucinations([]) == []

    def test_normal_segments_kept(self) -> None:
        segs = [
            _seg(0.0, 1.0, "Bonjour."),
            _seg(1.5, 3.0, "Comment ça va ?"),
            _seg(3.5, 5.0, "Très bien merci."),
        ]
        result = drop_whisper_hallucinations(segs)
        assert len(result) == 3

    def test_zero_duration_segment_dropped(self) -> None:
        """A segment with duration ≈ 0 ms is always a hallucination."""
        segs = [
            _seg(0.0, 1.0, "Real segment."),
            _seg(1.0, 1.0, "Zero duration"),  # 0 ms — dropped
            _seg(1.5, 2.5, "Another real one."),
        ]
        result = drop_whisper_hallucinations(segs)
        assert len(result) == 2
        assert "Zero duration" not in {s.text for s in result}

    def test_sub_100ms_segment_dropped(self) -> None:
        """Below the 100 ms duration floor, segments are dropped regardless of text."""
        segs = [
            _seg(0.0, 1.0, "A"),
            _seg(1.0, 1.05, "B"),  # 50 ms — dropped
            _seg(1.5, 2.5, "C"),
        ]
        result = drop_whisper_hallucinations(segs)
        assert [s.text for s in result] == ["A", "C"]

    def test_short_duration_verbatim_repeat_dropped(self) -> None:
        """A short (<1.5s) segment whose text matches a recent one is dropped."""
        segs = [
            _seg(0.0, 1.0, "Sans précipiter."),
            _seg(1.0, 1.5, "Sans précipiter."),  # 500 ms repeat — dropped
            _seg(2.0, 4.0, "Pour les ressources."),
        ]
        result = drop_whisper_hallucinations(segs)
        assert len(result) == 2
        assert result[0].text == "Sans précipiter."
        assert result[1].text == "Pour les ressources."

    def test_long_duration_verbatim_repeat_kept(self) -> None:
        """A long (≥1.5s) verbatim repeat is treated as legitimate (rhetorical
        emphasis, not a hallucination)."""
        segs = [
            _seg(0.0, 2.0, "Vraiment."),
            _seg(2.5, 4.5, "Vraiment."),  # 2.0 s — kept (rhetorical repeat)
        ]
        result = drop_whisper_hallucinations(segs)
        assert len(result) == 2

    def test_short_segment_with_unique_text_kept(self) -> None:
        """A short segment with text not seen recently is real speech."""
        segs = [
            _seg(0.0, 2.0, "Voici un long segment."),
            _seg(2.0, 2.3, "Bref."),  # 300 ms but unique — kept
        ]
        result = drop_whisper_hallucinations(segs)
        assert len(result) == 2

    def test_hallucination_pattern_from_real_data(self) -> None:
        """Mirror the actual Whisper output from feedback-ace.mp4 around 86–91s.

        Whisper produces 5 segments where 3 are hallucinations:
        - [89.4, 89.4] zero-duration repeat ×2
        - [90.7, 90.9] 200ms repeat of the prior 'Sans précipiter.'

        After filtering, only the 2 real segments should remain.
        """
        segs = [
            _seg(86.7, 89.3, "mais ce rythme-là m'a permis de vraiment pratiquer."),
            _seg(89.4, 89.4, "C'est faisable plus rapidement,"),  # 0 ms — dropped
            _seg(89.4, 89.4, "mais ce rythme-là m'a permis de vraiment pratiquer."),  # 0 ms — dropped
            _seg(89.5, 90.6, "Sans précipiter."),
            _seg(90.7, 90.9, "Sans précipiter."),  # 200 ms verbatim — dropped
        ]
        result = drop_whisper_hallucinations(segs)
        assert len(result) == 2
        assert result[0].text == "mais ce rythme-là m'a permis de vraiment pratiquer."
        assert result[1].text == "Sans précipiter."

    def test_drop_emits_warning_log(self, caplog) -> None:
        """Each drop is logged at WARNING with the timestamp and text."""
        segs = [
            _seg(0.0, 1.0, "Real"),
            _seg(1.0, 1.0, "Halluc"),  # zero-duration
        ]
        with caplog.at_level(logging.WARNING, logger="subtitle_studio.generate.transcribe"):
            drop_whisper_hallucinations(segs)
        assert any("Halluc" in rec.message for rec in caplog.records)

    def test_filter_does_not_mutate_input(self) -> None:
        segs = [
            _seg(0.0, 1.0, "A"),
            _seg(1.0, 1.0, "B"),
        ]
        before = list(segs)
        drop_whisper_hallucinations(segs)
        assert segs == before
