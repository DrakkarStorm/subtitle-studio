"""Unit tests for generate/subtitle.py formatters and line-wrapping."""

from types import SimpleNamespace

import srt as srt_lib

from subtitle_studio.generate.subtitle import MAX_CHARS, _vtt_time, to_srt, to_subtitles, to_vtt, wrap_text

# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------


class TestWrapText:
    def test_short_text_unchanged(self) -> None:
        assert wrap_text("Bonjour le monde.") == "Bonjour le monde."

    def test_long_line_wraps_at_max_chars(self) -> None:
        text = "Voici une phrase vraiment très longue qui dépasse la limite de cinquante caractères normalement."
        result = wrap_text(text)
        for line in result.splitlines():
            assert len(line) <= MAX_CHARS, f"Line too long: {line!r}"

    def test_max_two_lines(self) -> None:
        text = "Un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize."
        result = wrap_text(text)
        assert result.count("\n") <= 1, "More than 2 lines produced"

    def test_strips_whitespace(self) -> None:
        assert wrap_text("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert wrap_text("") == ""

    def test_exactly_max_chars_stays_one_line(self) -> None:
        text = "a" * MAX_CHARS
        assert wrap_text(text) == text

    def test_max_chars_32_wraps_at_32(self) -> None:
        text = "Voici une phrase qui dépasse trente-deux caractères facilement."
        result = wrap_text(text, max_chars=32)
        for line in result.splitlines():
            assert len(line) <= 32, f"Line too long: {line!r}"

    def test_max_chars_32_default_unchanged(self) -> None:
        text = "a" * MAX_CHARS
        assert wrap_text(text) == text  # default unchanged

    def test_exactly_32_chars_stays_one_line(self) -> None:
        text = "a" * 32
        assert wrap_text(text, max_chars=32) == text

    def test_max_chars_zero_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_chars"):
            wrap_text("hello", max_chars=0)

    def test_max_chars_negative_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_chars"):
            wrap_text("hello", max_chars=-5)

    def test_max_chars_nine_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="max_chars"):
            wrap_text("hello", max_chars=9)

    def test_max_chars_ten_does_not_raise(self) -> None:
        assert wrap_text("hello", max_chars=10) == "hello"


# ---------------------------------------------------------------------------
# _vtt_time
# ---------------------------------------------------------------------------


class TestVttTime:
    def test_zero(self) -> None:
        assert _vtt_time(0.0) == "00:00:00.000"

    def test_one_hour_30_minutes(self) -> None:
        assert _vtt_time(5400.5) == "01:30:00.500"

    def test_milliseconds_truncated_not_rounded(self) -> None:
        assert _vtt_time(1.999) == "00:00:01.999"


# ---------------------------------------------------------------------------
# to_srt
# ---------------------------------------------------------------------------


def _make_result(*segments):  # type: ignore[no-untyped-def]
    """Build a minimal fake stable-ts result object."""
    segs = [SimpleNamespace(start=s, end=e, text=t) for s, e, t in segments]
    return SimpleNamespace(segments=segs)


class TestToSrt:
    def test_basic_output(self) -> None:
        result = _make_result((0.0, 2.5, "Bonjour."), (3.0, 5.0, "Comment ça va ?"))
        srt_text = to_srt(result)
        assert "1\n" in srt_text
        assert "2\n" in srt_text
        assert "Bonjour." in srt_text
        assert "Comment ça va ?" in srt_text

    def test_srt_timestamp_format(self) -> None:
        result = _make_result((0.0, 2.5, "Hello"))
        srt_text = to_srt(result)
        assert "00:00:00,000 --> 00:00:02,500" in srt_text

    def test_line_wrapping_applied(self) -> None:
        long_text = (
            "Voici une phrase vraiment très longue qui dépasse largement la limite "
            "autorisée par défaut sans aucun doute possible."
        )
        result = _make_result((0.0, 5.0, long_text))
        srt_text = to_srt(result)
        lines = srt_text.strip().splitlines()
        content_lines = lines[2:]
        for line in content_lines:
            assert len(line) <= MAX_CHARS

    def test_empty_segments(self) -> None:
        result = _make_result()
        assert to_srt(result).strip() == ""

    def test_max_chars_32_wraps_content_at_32(self) -> None:
        long_text = "Voici une phrase vraiment très longue qui dépasse trente-deux caractères."
        result = _make_result((0.0, 5.0, long_text))
        srt_text = to_srt(result, max_chars=32)
        lines = srt_text.strip().splitlines()
        content_lines = [ln for ln in lines if ln and "-->" not in ln and not ln.strip().isdigit()]
        for line in content_lines:
            assert len(line) <= 32, f"Line too long: {line!r}"

    def test_default_max_chars_keeps_short_text_intact(self) -> None:
        text = "a" * 35  # fits in default MAX_CHARS, not in 32
        result = _make_result((0.0, 2.0, text))
        srt_text = to_srt(result)  # default
        assert text in srt_text  # not wrapped


# ---------------------------------------------------------------------------
# to_subtitles
# ---------------------------------------------------------------------------


class TestToSubtitles:
    def test_returns_list_matching_segment_count(self) -> None:
        result = _make_result(
            (0.0, 2.0, "Un"),
            (2.0, 4.0, "deux"),
            (4.0, 6.0, "trois"),
        )
        subs = to_subtitles(result)
        assert len(subs) == 3
        assert all(isinstance(s, srt_lib.Subtitle) for s in subs)

    def test_empty_segments_returns_empty_list(self) -> None:
        result = _make_result()
        assert to_subtitles(result) == []

    def test_wraps_content_per_max_chars(self) -> None:
        # Use separated words so textwrap can break between them
        # (wrap_text uses break_long_words=False by design).
        long_text = "word " * 12  # 60 chars, splittable
        result = _make_result((0.0, 2.0, long_text))
        subs = to_subtitles(result, max_chars=32)
        assert len(subs) == 1
        for line in subs[0].content.splitlines():
            assert len(line) <= 32

    def test_to_srt_equals_compose_of_to_subtitles(self) -> None:
        """Regression: to_srt must stay byte-equivalent to compose(to_subtitles)."""
        result = _make_result((0.0, 2.0, "Bonjour."), (3.0, 5.0, "Comment ça va ?"))
        assert to_srt(result) == srt_lib.compose(to_subtitles(result))


# ---------------------------------------------------------------------------
# to_vtt
# ---------------------------------------------------------------------------


class TestToVtt:
    def test_starts_with_webvtt(self) -> None:
        result = _make_result((0.0, 1.0, "Salut"))
        assert to_vtt(result).startswith("WEBVTT")

    def test_vtt_timestamp_format(self) -> None:
        result = _make_result((0.0, 2.5, "Salut"))
        vtt_text = to_vtt(result)
        assert "00:00:00.000 --> 00:00:02.500" in vtt_text

    def test_sequential_indices(self) -> None:
        result = _make_result((0.0, 1.0, "Un"), (1.5, 2.5, "Deux"), (3.0, 4.0, "Trois"))
        vtt_text = to_vtt(result)
        assert "\n1\n" in vtt_text
        assert "\n2\n" in vtt_text
        assert "\n3\n" in vtt_text

    def test_line_wrapping_applied(self) -> None:
        long_text = (
            "Voici une phrase vraiment très longue qui dépasse largement la limite "
            "autorisée par défaut sans aucun doute possible."
        )
        result = _make_result((0.0, 5.0, long_text))
        vtt_text = to_vtt(result)
        lines = vtt_text.splitlines()
        content_lines = [
            line
            for line in lines
            if line and not line.startswith("WEBVTT") and "-->" not in line and not line.isdigit()
        ]
        for line in content_lines:
            assert len(line) <= MAX_CHARS
