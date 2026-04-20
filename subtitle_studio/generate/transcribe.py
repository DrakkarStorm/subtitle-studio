"""Transcription dispatcher — local faster-whisper vs OpenAI-compatible cloud API."""

from pathlib import Path
from typing import Any


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


def _transcribe_local(audio_path: Path, model_size: str, device: str) -> Any:
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
    )


def transcribe(
    audio_path: Path,
    model_size: str = "large-v3",
    device: str = "auto",
    backend: str = "local",
    api_url: str | None = None,
    api_key: str | None = None,
    api_model: str = "whisper-large-v3",
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
    """
    if backend == "api":
        from .transcribe_cloud import transcribe_api

        return transcribe_api(
            audio_path,
            base_url=api_url,
            api_key=api_key,
            model=api_model,
        )
    return _transcribe_local(audio_path, model_size, device)
