"""Tests for detect/detector (pure functions + mocked API calls)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from subtitle_studio.detect.detector import (
    build_system_prompt,
    build_user_prompt,
    call_claude,
    detect_errors,
    load_branding,
)
from subtitle_studio.detect.models import BrandingConfig, ClaudeAPIError
from tests.detect.conftest import make_claude_response

# ---------------------------------------------------------------------------
# Pure function tests (no mocks)
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_contains_branding_names(self, sample_branding: BrandingConfig) -> None:
        prompt = build_system_prompt("Kubernetes tutorial", sample_branding)
        assert "Alice" in prompt
        assert "Taverne Tech" in prompt

    def test_contains_branding_vocab(self, sample_branding: BrandingConfig) -> None:
        prompt = build_system_prompt("", sample_branding)
        assert "Kubernetes" in prompt
        assert "Docker" in prompt

    def test_contains_context(self, sample_branding: BrandingConfig) -> None:
        prompt = build_system_prompt("Docker tutorial for beginners", sample_branding)
        assert "Docker tutorial for beginners" in prompt

    def test_empty_context_uses_default(self, sample_branding: BrandingConfig) -> None:
        # The default fallback is intentionally in French (part of the LLM prompt).
        prompt = build_system_prompt("", sample_branding)
        assert "Non précisé" in prompt

    def test_contains_untrusted_data_instruction(self, sample_branding: BrandingConfig) -> None:
        prompt = build_system_prompt("", sample_branding)
        assert "non fiable" in prompt.lower() or "TEXTE NON FIABLE" in prompt


class TestBuildUserPrompt:
    def test_uses_xml_segment_delimiters(self, sample_subtitles: list) -> None:
        prompt = build_user_prompt(sample_subtitles)
        assert '<segment id="1">' in prompt
        assert '<segment id="2">' in prompt
        assert "</segment>" in prompt

    def test_includes_subtitle_content(self, sample_subtitles: list) -> None:
        prompt = build_user_prompt(sample_subtitles)
        assert sample_subtitles[0].content in prompt

    def test_single_subtitle(self, sample_subtitles: list) -> None:
        prompt = build_user_prompt([sample_subtitles[0]])
        assert '<segment id="1">' in prompt
        assert '<segment id="2">' not in prompt


# ---------------------------------------------------------------------------
# call_claude tests (with mocks)
# ---------------------------------------------------------------------------


class TestCallClaude:
    def test_valid_response_returns_corrections(self, mock_anthropic_client: MagicMock, sample_subtitles: list) -> None:
        payload = [
            {
                "segment": 1,
                "original": "je suis aller",
                "suggestion": "je suis allé",
                "raison": "Participe passé.",
            }
        ]
        mock_anthropic_client.messages.create.return_value = make_claude_response(payload)

        system = "system prompt"
        user_msg = build_user_prompt(sample_subtitles[:1])
        corrections = call_claude(mock_anthropic_client, system, user_msg, sample_subtitles[:1])

        assert len(corrections) == 1
        assert corrections[0].segment == 1
        assert corrections[0].suggestion == "je suis allé"

    def test_empty_response_returns_empty_list(self, mock_anthropic_client: MagicMock, sample_subtitles: list) -> None:
        mock_anthropic_client.messages.create.return_value = make_claude_response([])
        corrections = call_claude(mock_anthropic_client, "system", "user", sample_subtitles[:1])
        assert corrections == []

    def test_malformed_json_returns_empty_list(self, mock_anthropic_client: MagicMock, sample_subtitles: list) -> None:
        content_block = MagicMock()
        content_block.text = "Sorry, I don't understand."
        mock_anthropic_client.messages.create.return_value = MagicMock(content=[content_block])
        corrections = call_claude(mock_anthropic_client, "system", "user", sample_subtitles[:1])
        assert corrections == []

    def test_out_of_range_segment_ignored(self, mock_anthropic_client: MagicMock, sample_subtitles: list) -> None:
        payload = [{"segment": 999, "original": "x", "suggestion": "y", "raison": "test"}]
        mock_anthropic_client.messages.create.return_value = make_claude_response(payload)
        corrections = call_claude(mock_anthropic_client, "system", "user", sample_subtitles)
        assert corrections == []

    def test_markdown_fences_stripped(self, mock_anthropic_client: MagicMock, sample_subtitles: list) -> None:
        payload = [{"segment": 1, "original": "x", "suggestion": "y", "raison": "test"}]
        raw_with_fence = "```json\n" + json.dumps(payload)[1:] + "\n```"
        content_block = MagicMock()
        content_block.text = raw_with_fence
        mock_anthropic_client.messages.create.return_value = MagicMock(content=[content_block])
        corrections = call_claude(mock_anthropic_client, "system", "user", sample_subtitles[:1])
        assert len(corrections) == 1

    def test_auth_error_raises_claude_api_error(self, mock_anthropic_client: MagicMock, sample_subtitles: list) -> None:
        import anthropic as _anthropic

        mock_anthropic_client.messages.create.side_effect = _anthropic.AuthenticationError(
            message="Invalid key", response=MagicMock(), body={}
        )
        with pytest.raises(ClaudeAPIError, match="Invalid"):
            call_claude(mock_anthropic_client, "system", "user", sample_subtitles[:1])


# ---------------------------------------------------------------------------
# load_branding tests
# ---------------------------------------------------------------------------


class TestLoadBranding:
    def test_loads_valid_branding(self, tmp_path: Path) -> None:
        yaml_content = """
noms_propres:
  - Alice
  - Taverne Tech
vocabulaire_technique:
  - Kubernetes
  - Docker
"""
        path = tmp_path / "branding.yaml"
        path.write_text(yaml_content, encoding="utf-8")
        branding = load_branding(path)
        assert "Alice" in branding.noms_propres
        assert "Kubernetes" in branding.vocabulaire_technique

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "branding.yaml"
        path.write_text("{ invalid yaml :", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid"):
            load_branding(path)

    def test_missing_fields_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "branding.yaml"
        path.write_text("noms_propres:\n  - Alice\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_branding(path)


# ---------------------------------------------------------------------------
# detect_errors tests (orchestrator)
# ---------------------------------------------------------------------------


class TestDetectErrors:
    def test_aggregates_corrections_across_batches(
        self,
        mock_anthropic_client: MagicMock,
        sample_subtitles: list,
        sample_branding: BrandingConfig,
    ) -> None:
        payload = [{"segment": 1, "original": "aller", "suggestion": "allé", "raison": "participe"}]
        mock_anthropic_client.messages.create.return_value = make_claude_response(payload)

        corrections = detect_errors(
            sample_subtitles,
            "Kubernetes tutorial",
            sample_branding,
            mock_anthropic_client,
            batch_size=2,
        )

        assert mock_anthropic_client.messages.create.call_count >= 1
        assert any(c.segment == 1 for c in corrections)

    def test_on_batch_callback_called_per_batch(
        self,
        mock_anthropic_client: MagicMock,
        sample_subtitles: list,
        sample_branding: BrandingConfig,
    ) -> None:
        mock_anthropic_client.messages.create.return_value = make_claude_response([])
        call_count = 0

        def on_batch() -> None:
            nonlocal call_count
            call_count += 1

        detect_errors(
            sample_subtitles,
            "",
            sample_branding,
            mock_anthropic_client,
            batch_size=2,
            on_batch=on_batch,
        )

        assert call_count == 2
