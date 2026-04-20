"""ffmpeg-based audio extraction for subtitle transcription."""

import subprocess
import tempfile
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}


def extract_audio(video_path: Path) -> Path:
    """Extract audio from a video file to a temporary 16kHz mono MP3 at 32 kbps.

    Raises:
        ValueError: If the file extension is not supported, or the video has no audio track.
        FileNotFoundError: If ffmpeg is not on PATH.
        subprocess.CalledProcessError: If ffmpeg fails (e.g. corrupt file).
    """
    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: '{video_path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)  # noqa: SIM115
    tmp.close()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-b:a",
                "32k",
                tmp.name,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        if "does not contain any stream" in stderr:
            raise ValueError("The video has no audio track. Cannot generate subtitles.") from e
        raise

    return Path(tmp.name)
