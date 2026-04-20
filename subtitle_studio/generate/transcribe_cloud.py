"""OpenAI-compatible Whisper cloud client (vendor-neutral)."""

import os
from pathlib import Path
from typing import Any, NoReturn

_SIZE_LIMIT_BYTES = 25 * 1024 * 1024  # 25 MB


def transcribe_api(
    audio_path: Path,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str = "whisper-large-v3",
    language: str = "fr",
) -> Any:
    """Transcribe audio via any OpenAI-compatible Whisper endpoint.

    Returns a TranscriptionVerbose object whose .segments list contains objects
    with .start, .end, and .text — duck-type compatible with generate.subtitle.to_srt().

    Raises:
        RuntimeError: on API errors (auth, rate limit, file too large, network)
                      or if openai package is not installed.
        ValueError:   if the audio file exceeds the 25 MB limit.
    """
    try:
        from openai import APIConnectionError, APIStatusError, OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package required for --backend api. Install with: pip install 'subtitle-studio[api]'"
        ) from e

    file_size = audio_path.stat().st_size
    if file_size > _SIZE_LIMIT_BYTES:
        mb = file_size / 1024 / 1024
        raise ValueError(f"Audio too large for API ({mb:.1f} MB > 25 MB limit). Use --backend local for long videos.")

    resolved_url = base_url or os.environ.get("WHISPER_API_URL")
    if not resolved_url:
        raise RuntimeError(
            "API URL required: set WHISPER_API_URL to an OpenAI-compatible Whisper endpoint "
            "(e.g. https://api.openai.com/v1) or pass --api-url."
        )
    resolved_key = api_key or os.environ.get("WHISPER_API_KEY")
    if not resolved_key:
        raise RuntimeError("API key required: set WHISPER_API_KEY or use --api-key")

    client = OpenAI(base_url=resolved_url, api_key=resolved_key)

    try:
        with audio_path.open("rb") as f:
            result = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                temperature=0.0,
            )
    except APIConnectionError as e:
        raise RuntimeError(f"Could not reach API endpoint: {resolved_url}") from e
    except APIStatusError as e:
        _raise_for_status(e)
    else:
        if result.segments is None:
            raise RuntimeError(
                "Cloud transcription returned no segments. "
                "Ensure the endpoint supports verbose_json with segment timestamps."
            )
        return result


def _raise_for_status(e: object) -> NoReturn:
    from openai import APIStatusError

    assert isinstance(e, APIStatusError)
    code = e.status_code
    if code == 401:
        raise RuntimeError("API authentication failed. Check your API key.")
    if code == 413:
        raise RuntimeError("File rejected by API (too large).")
    if code == 429:
        raise RuntimeError("API rate limit exceeded. Retry later.")
    raise RuntimeError(f"API error {code}: {e.message}")
