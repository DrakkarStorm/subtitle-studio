"""Transcription dispatcher — local faster-whisper vs OpenAI-compatible cloud API."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Anti-hallucination filter — defaults tuned for the
# ``condition_on_previous_text=True`` failure mode where Whisper produces
# zero-duration repeats of recent text on hesitations/pauses.
_HALLUCINATION_MIN_DURATION_S = 0.1  # below this, a segment is impossible (real speech)
_HALLUCINATION_MATCH_WINDOW_S = 1.5  # short-duration repeats checked against this window
_HALLUCINATION_RECENT_LOOKBACK = 3  # how many previous segments to check for exact match


def drop_whisper_hallucinations(segments: list[Any]) -> list[Any]:
    """Drop Whisper segments that match known hallucination signatures.

    Two heuristics, applied in order to each segment:

    1. **Zero / quasi-zero duration** (< :data:`_HALLUCINATION_MIN_DURATION_S`)
       — real speech cannot fit in <100 ms; this is a deterministic Whisper
       artefact triggered by ``condition_on_previous_text=True``.
    2. **Short-duration verbatim repeat**: duration <
       :data:`_HALLUCINATION_MATCH_WINDOW_S` *and* the text matches one of the
       previous :data:`_HALLUCINATION_RECENT_LOOKBACK` kept segments verbatim.
       A human speaker does not re-emit the same exact words in under 1.5 s.

    Both signals are documented Whisper failure modes. Each drop is logged at
    WARNING with the timestamp + text so the operator can audit decisions.

    Returns a new list — does not mutate the input.
    """
    cleaned: list[Any] = []
    for seg in segments:
        duration = seg.end - seg.start
        seg_text = seg.text.strip()

        if duration < _HALLUCINATION_MIN_DURATION_S:
            logger.warning(
                "Dropped Whisper hallucination at %.2fs (%dms): %r",
                seg.start,
                int(duration * 1000),
                seg_text,
            )
            continue

        if duration < _HALLUCINATION_MATCH_WINDOW_S:
            recent_texts = {s.text.strip() for s in cleaned[-_HALLUCINATION_RECENT_LOOKBACK:]}
            if seg_text in recent_texts:
                logger.warning(
                    "Dropped repeated Whisper segment at %.2fs (%dms): %r",
                    seg.start,
                    int(duration * 1000),
                    seg_text,
                )
                continue

        cleaned.append(seg)
    return cleaned


def _resolve_device(device: str) -> tuple[str, str]:
    """Resolve 'auto' to 'cuda' or 'cpu' and return (device, compute_type)."""
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def _transcribe_local(
    audio_path: Path,
    model_size: str,
    device: str,
    initial_prompt: str | None = None,
) -> Any:
    """Transcribe using the local faster-whisper model via stable-ts."""
    import stable_whisper  # lazy import — avoids loading ~3 GB at package startup

    resolved_device, compute_type = _resolve_device(device)
    model = stable_whisper.load_faster_whisper(
        model_size,
        device=resolved_device,
        compute_type=compute_type,
    )
    return model.transcribe(
        str(audio_path),
        language="fr",
        beam_size=5,
        vad=True,
        suppress_silence=True,
        initial_prompt=initial_prompt,
    )


def transcribe(
    audio_path: Path,
    model_size: str = "large-v3",
    device: str = "auto",
    backend: str = "local",
    api_url: str | None = None,
    api_key: str | None = None,
    api_model: str = "whisper-large-v3",
    initial_prompt: str | None = None,
) -> Any:
    """Transcribe French audio, dispatching to local or cloud backend.

    Returns an object with .segments iterable (each segment: .start, .end, .text).

    Args:
        audio_path:  Path to the audio file (16 kHz mono MP3 recommended).
        model_size:  Local faster-whisper model size (local backend only).
                     Accepted: 'tiny' (~75 MB), 'base' (~140 MB), 'small' (~460 MB),
                     'medium' (~1.5 GB), 'large-v2', 'large-v3' (~3 GB, default).
                     The first use downloads from HuggingFace Hub into
                     ~/.cache/huggingface/hub/. 'large-v3' is recommended for French
                     quality on machines with ≥16 GB RAM.
        device:      Inference device — 'auto' (detects CUDA, falls back to CPU),
                     'cpu', 'cuda' (local backend only). On macOS there is no GPU
                     acceleration: 'auto' and 'cpu' are equivalent.
        backend:     'local' (default) or 'api'.
        api_url:     OpenAI-compatible base URL (api backend only).
        api_key:     Bearer token for the API (api backend only).
        api_model:   Model name to request from the endpoint (api backend only).
        initial_prompt: Optional vocabulary prompt biasing the transcription
                     toward known terms (proper nouns, technical jargon).
                     Whisper uses it as the seed of the previous-context window;
                     keep it short (<224 tokens) — typically the comma-separated
                     branding list.
    """
    if backend == "api":
        from .transcribe_cloud import transcribe_api

        return transcribe_api(
            audio_path,
            base_url=api_url,
            api_key=api_key,
            model=api_model,
            initial_prompt=initial_prompt,
        )
    return _transcribe_local(audio_path, model_size, device, initial_prompt=initial_prompt)
