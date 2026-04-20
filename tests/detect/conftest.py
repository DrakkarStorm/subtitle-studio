"""Shared test fixtures."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import srt

from subtitle_studio.detect.models import BrandingConfig, Correction


@pytest.fixture
def sample_subtitles() -> list[srt.Subtitle]:
    """3 test subtitles, including one with a classic ASR error."""
    return [
        srt.Subtitle(
            index=1,
            start=datetime.timedelta(seconds=1),
            end=datetime.timedelta(seconds=3),
            content="Bonjour, je suis aller au marché.",
        ),
        srt.Subtitle(
            index=2,
            start=datetime.timedelta(seconds=4),
            end=datetime.timedelta(seconds=6),
            content="Il fait beau aujourd'hui.",
        ),
        srt.Subtitle(
            index=3,
            start=datetime.timedelta(seconds=7),
            end=datetime.timedelta(seconds=9),
            content="ça c'est passer comme ça",
        ),
    ]


@pytest.fixture
def sample_srt_file(tmp_path: Path, sample_subtitles: list[srt.Subtitle]) -> Path:
    """Temporary SRT file containing the sample_subtitles."""
    path = tmp_path / "test.srt"
    path.write_text(srt.compose(sample_subtitles), encoding="utf-8")
    return path


@pytest.fixture
def sample_branding() -> BrandingConfig:
    """Minimal branding fixture for tests."""
    return BrandingConfig(
        noms_propres=["Alice", "Taverne Tech"],
        vocabulaire_technique=["Kubernetes", "Docker", "CI/CD"],
    )


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mocked Anthropic client."""
    return MagicMock()


def make_claude_response(payload: list[dict[str, Any]]) -> MagicMock:
    """Build a fake Claude API response (messages.create format)."""
    content_block = MagicMock()
    content_block.text = json.dumps(payload)[1:]  # strip the leading "[" (prefill)
    response = MagicMock()
    response.content = [content_block]
    return response


@pytest.fixture
def sample_correction() -> Correction:
    return Correction(
        segment=1,
        original="je suis aller au marché",
        suggestion="je suis allé au marché",
        raison="Confusion infinitif/participe passé",
    )
