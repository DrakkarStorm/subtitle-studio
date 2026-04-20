"""SRT subtitle translation from French to the target language via the Claude API."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import srt
from anthropic import Anthropic

from ..detect.models import ClaudeAPIError
from ..detect.srt_parser import parse_srt, write_srt
from ..generate.subtitle import MAX_CHARS, wrap_text

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

SUPPORTED_LANGS: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
}


def translate_cues(
    subtitles: list[srt.Subtitle],
    client: Anthropic,
    model: str = MODEL,
    max_chars: int = MAX_CHARS,
    target_lang: str = "en",
) -> list[srt.Subtitle]:
    """Translate a list of French subtitles via Claude.

    Segments are sent in a single prompt using the numbered format `[N] text`
    (temperature 0 for deterministic 1→1 mapping). Original timings are
    preserved; only the text content is replaced. Translated text is
    re-wrapped with `wrap_text(max_chars)` to respect line conventions.

    Args:
        subtitles: Source subtitles (FR), with timings.
        client: Already-instantiated Anthropic client.
        model: Claude model ID (default: `MODEL`).
        max_chars: Maximum line width for wrapping (default: 42).
        target_lang: Target language code — must be a key of `SUPPORTED_LANGS`
            (`en`, `es`, `de`, `pt`). An unsupported code raises `KeyError`.

    Returns:
        List of translated subtitles, same length as `subtitles`, same indices
        and same timings. Segments the LLM did not return (partial parse) keep
        the **original French text** — only a `logger.warning` is emitted with
        the missing indices.

    Raises:
        ClaudeAPIError: If the API call fails, if the response format is
            invalid, or if the response was truncated
            (`stop_reason == "max_tokens"`).
    """
    lang_name = SUPPORTED_LANGS[target_lang]
    numbered = "\n".join(f"[{i + 1}] {sub.content}" for i, sub in enumerate(subtitles))

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            system=(
                f"You are translating French YouTube video subtitles to {lang_name}.\n"
                "Rules:\n"
                "- Translate naturally and fluently, not word-for-word\n"
                "- Keep the same tone and energy as the original\n"
                "- Return ONLY the translations, one per line, in the exact same [N] format\n"
                "- Do not add any explanation, commentary, or extra text"
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"French subtitles:\n{numbered}\n\n{lang_name} translations:",
                }
            ],
        )
    except Exception as exc:
        raise ClaudeAPIError(f"Claude API error during translation: {exc}") from exc

    if message.stop_reason == "max_tokens":
        raise ClaudeAPIError(
            "Claude response truncated (max_tokens reached) during translation. "
            "Retry with fewer segments or increase max_tokens."
        )

    if not message.content or message.content[0].type != "text":
        raise ClaudeAPIError("Unexpected response format from the Claude API.")

    response_text = message.content[0].text.strip()

    translations = [""] * len(subtitles)
    for line in response_text.splitlines():
        match = re.match(r"^\[(\d+)\]\s*(.*)", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(subtitles):
                translations[idx] = match.group(2)

    missing = [i + 1 for i, t in enumerate(translations) if not t]
    if missing:
        logger.warning(
            "%d subtitle(s) not translated (indices: %s). Original text preserved.",
            len(missing),
            missing,
        )

    result: list[srt.Subtitle] = []
    for sub, translated in zip(subtitles, translations, strict=True):
        content = wrap_text(translated, max_chars) if translated else sub.content
        result.append(
            srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=content,
            )
        )
    return result


def run_translation(
    input_srt: Path,
    output_dir: Path,
    client: Anthropic,
    model: str = MODEL,
    max_chars: int = MAX_CHARS,
    target_lang: str = "en",
) -> Path:
    """Orchestrate parse → translate → write.

    The output file is named `<stem>.<target_lang>.srt` in `output_dir`
    (e.g. `video.srt` + `target_lang="en"` → `video.en.srt`). Writes are atomic
    (see `srt_parser.write_srt`).

    Raises:
        SRTParseError: If `input_srt` is missing or unreadable.
        ClaudeAPIError: Propagated from `translate_cues` on API failure.
    """
    subtitles = parse_srt(input_srt)
    translated = translate_cues(subtitles, client, model, max_chars=max_chars, target_lang=target_lang)
    output_path = output_dir / (input_srt.stem + f".{target_lang}.srt")
    write_srt(translated, output_path)
    return output_path
