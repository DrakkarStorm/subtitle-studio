"""Tests for translate/translate.py."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import srt

from subtitle_studio.detect.models import ClaudeAPIError
from subtitle_studio.translate.translate import SUPPORTED_LANGS, run_translation, translate_cues


def _make_translate_response(translations: list[str]) -> MagicMock:
    """Build a fake Claude API response for translation."""
    text = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(translations))
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def sample_subtitles() -> list[srt.Subtitle]:
    return [
        srt.Subtitle(
            index=1,
            start=datetime.timedelta(seconds=1),
            end=datetime.timedelta(seconds=3),
            content="Bonjour le monde.",
        ),
        srt.Subtitle(
            index=2,
            start=datetime.timedelta(seconds=4),
            end=datetime.timedelta(seconds=6),
            content="Comment ça va ?",
        ),
    ]


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _make_translate_response(["Hello world.", "How are you?"])
    return client


# ---------------------------------------------------------------------------
# Tests SUPPORTED_LANGS
# ---------------------------------------------------------------------------


class TestSupportedLangs:
    def test_contains_en(self) -> None:
        assert "en" in SUPPORTED_LANGS
        assert SUPPORTED_LANGS["en"] == "English"

    def test_contains_es(self) -> None:
        assert "es" in SUPPORTED_LANGS
        assert SUPPORTED_LANGS["es"] == "Spanish"

    def test_contains_de(self) -> None:
        assert "de" in SUPPORTED_LANGS
        assert SUPPORTED_LANGS["de"] == "German"

    def test_contains_pt(self) -> None:
        assert "pt" in SUPPORTED_LANGS
        assert SUPPORTED_LANGS["pt"] == "Portuguese"


# ---------------------------------------------------------------------------
# Tests translate_cues
# ---------------------------------------------------------------------------


class TestTranslateCues:
    def test_default_lang_en_system_prompt(self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock) -> None:
        translate_cues(sample_subtitles, mock_client)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "English" in call_kwargs["system"]

    def test_target_lang_es_system_prompt(self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock) -> None:
        translate_cues(sample_subtitles, mock_client, target_lang="es")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "Spanish" in call_kwargs["system"]
        assert "English" not in call_kwargs["system"]

    def test_target_lang_es_user_message_label(
        self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock
    ) -> None:
        translate_cues(sample_subtitles, mock_client, target_lang="es")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "Spanish translations:" in user_content

    def test_default_lang_en_user_message_label(
        self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock
    ) -> None:
        translate_cues(sample_subtitles, mock_client)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "English translations:" in user_content

    def test_returns_translated_subtitles(self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock) -> None:
        result = translate_cues(sample_subtitles, mock_client)
        assert len(result) == 2
        assert result[0].content == "Hello world."
        assert result[1].content == "How are you?"

    def test_temperature_zero_sent(self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock) -> None:
        translate_cues(sample_subtitles, mock_client)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    def test_unsupported_lang_raises_key_error(
        self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock
    ) -> None:
        """translate_cues() with an unknown language raises KeyError — documented behavior."""
        with pytest.raises(KeyError):
            translate_cues(sample_subtitles, mock_client, target_lang="zh")

    def test_max_tokens_truncation_raises(self, sample_subtitles: list[srt.Subtitle], mock_client: MagicMock) -> None:
        """A truncated response (stop_reason=max_tokens) must raise ClaudeAPIError."""
        response = _make_translate_response(["Hello world."])
        response.stop_reason = "max_tokens"
        mock_client.messages.create.return_value = response

        with pytest.raises(ClaudeAPIError, match="truncated"):
            translate_cues(sample_subtitles, mock_client)


# ---------------------------------------------------------------------------
# Tests run_translation
# ---------------------------------------------------------------------------


class TestRunTranslation:
    def test_output_filename_en(self, tmp_path: Path, mock_client: MagicMock) -> None:
        input_srt = tmp_path / "video.srt"
        subtitles = [
            srt.Subtitle(
                index=1,
                start=datetime.timedelta(seconds=1),
                end=datetime.timedelta(seconds=3),
                content="Bonjour.",
            )
        ]
        input_srt.write_text(srt.compose(subtitles), encoding="utf-8")
        mock_client.messages.create.return_value = _make_translate_response(["Hello."])

        result = run_translation(input_srt, tmp_path, mock_client)

        assert result.name == "video.en.srt"
        assert result.exists()

    def test_output_filename_de(self, tmp_path: Path, mock_client: MagicMock) -> None:
        input_srt = tmp_path / "video.srt"
        subtitles = [
            srt.Subtitle(
                index=1,
                start=datetime.timedelta(seconds=1),
                end=datetime.timedelta(seconds=3),
                content="Bonjour.",
            )
        ]
        input_srt.write_text(srt.compose(subtitles), encoding="utf-8")
        mock_client.messages.create.return_value = _make_translate_response(["Hallo."])

        result = run_translation(input_srt, tmp_path, mock_client, target_lang="de")

        assert result.name == "video.de.srt"
        assert result.exists()

    def test_output_filename_corrected_stem(self, tmp_path: Path, mock_client: MagicMock) -> None:
        """<stem>_corrected.srt → <stem>_corrected.es.srt"""
        input_srt = tmp_path / "video_corrected.srt"
        subtitles = [
            srt.Subtitle(
                index=1,
                start=datetime.timedelta(seconds=1),
                end=datetime.timedelta(seconds=3),
                content="Bonjour.",
            )
        ]
        input_srt.write_text(srt.compose(subtitles), encoding="utf-8")
        mock_client.messages.create.return_value = _make_translate_response(["Hola."])

        result = run_translation(input_srt, tmp_path, mock_client, target_lang="es")

        assert result.name == "video_corrected.es.srt"
