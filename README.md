# subtitle-studio

Unified CLI pipeline for YouTube subtitles — extraction, verification and translation in a single command.

[![CI](https://github.com/DrakkarStorm/subtitle-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/DrakkarStorm/subtitle-studio/actions/workflows/ci.yml)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) — package manager
- [ffmpeg](https://ffmpeg.org/) — audio extraction

```bash
brew install ffmpeg
```

## Installation

### End user

```bash
uv tool install .
```

With the cloud Whisper backend:

```bash
uv tool install ".[api]"
```

### Developer

```bash
git clone <repo>
cd subtitle_studio
uv sync --group dev
```

## Configuration

Create a `.env` file at the project root (see `.env.example`):

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
```

Or export the variable in your shell:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | — | Claude API key |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5-20251001` | Claude model for the whole pipeline (equivalent to `--model`) |
| `BRANDING_YAML_PATH` | No | bundled in the package | Override the `branding.yaml` file |
| `WHISPER_API_URL` | No* | — | URL of the OpenAI-compatible Whisper endpoint |
| `WHISPER_API_KEY` | No* | — | Token for the cloud Whisper backend |

*Only required with `--backend api`.

## First run — model download

The Whisper `large-v3` model (~3 GB) is downloaded from HuggingFace Hub on the first run. To avoid a silent delay, pre-download the model:

```bash
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
```

The cache lives under `~/.cache/huggingface/hub/`. Subsequent runs use the local cache.

## Usage

```bash
subtitle-studio video.mp4
```

### Options

| Option | Default | Description |
|---|---|---|
| `video` | — | Video file (`.mp4`, `.mkv`, `.mov`, `.avi`) |
| `--context`, `-c` | `""` | Video topic (e.g. `"Kubernetes tutorial"`) — improves detection |
| `--backend`, `-b` | `local` | Whisper backend: `local` (GPU/CPU) or `api` (cloud) |
| `--output`, `-o` | `<stem>_YYYYMMDD_HHMMSS/` | Output directory |
| `--model`, `-m` | `claude-haiku-4-5-20251001` | Claude model used for both detection and translation (also `ANTHROPIC_MODEL`) |
| `--short`, `-s` | `false` | YouTube Shorts format (9:16 vertical) — strict CPS/CPL/duration enforcement, auto-fix, blocking errors on violations |
| `--target-lang`, `-l` | `en` | Target translation language: `en`, `es`, `de`, `pt` |
| `--check-guidelines` | `false` | Audit YouTube guidelines (CPS/CPL/duration/gap) in landscape mode. Read-only — emits warnings, never auto-fixes or blocks. No-op with `--short`. |
| `--max-cps` | `21.0` | CPS threshold (chars/s) — only applied with `--short` or `--check-guidelines` |
| `--warn-cps` | `18.0` | CPS warning threshold — only applied with `--short` or `--check-guidelines` |
| `--max-cpl` | `42` | CPL threshold (chars/line) — only applied with `--short` or `--check-guidelines` |
| `--warn-cpl` | `40` | CPL warning threshold — only applied with `--short` or `--check-guidelines` |
| `--min-gap` | `80` | Minimum gap between subtitles (ms) — only applied with `--short` or `--check-guidelines` |
| `--verbose`, `-v` | — | Increase log verbosity (`-v` = INFO, `-vv` = DEBUG) |
| `--quiet`, `-q` | — | Errors only (takes precedence over `--verbose`) |
| `--version`, `-V` | — | Print the version and exit |

### Examples

```bash
# Simple run
subtitle-studio video.mp4

# With topic context (recommended for technical vocabulary)
subtitle-studio video.mp4 --context "Kubernetes tutorial"

# Custom output directory
subtitle-studio video.mp4 --output /tmp/my_video/

# Cloud Whisper backend
subtitle-studio video.mp4 --backend api

# More powerful Claude model for the whole pipeline
subtitle-studio video.mp4 --model claude-sonnet-4-6

# YouTube Shorts format (9:16 vertical)
subtitle-studio video.mp4 --short

# Translate to Spanish instead of English
subtitle-studio video.mp4 --target-lang es

# Override default YouTube thresholds
subtitle-studio video.mp4 --max-cps 25 --warn-cps 22 --min-gap 100
```

## Pipeline and output artifacts

The pipeline chains 3 stages:

1. **Extraction** — ffmpeg extracts audio, Whisper transcribes, and (in landscape mode) segments are merged into YouTube-like phrases (~5 s / ~80 chars, preferring sentence boundaries).
2. **Verification** — ASR corrections by Claude. In `--short` mode, strict CPS/CPL/duration/gap enforcement runs with auto-fix and blocking errors. In landscape mode, no enforcement unless `--check-guidelines` is passed (read-only audit).
3. **Translation** — Claude translates the corrected subtitles to the target language (`--target-lang`: `en`, `es`, `de` or `pt`).

### Modes

| Mode | Triggered by | Segmentation | Guideline checks |
|---|---|---|---|
| **Landscape (default)** | (no flag) | Sentence-level merge — long segments ~5 s, capped at 2 lines × 42 chars | None, unless `--check-guidelines` |
| **Shorts** | `--short` | Raw Whisper segments, auto-merged and auto-split to respect CPS ≤ 21 / duration ≥ 0.5 s | Strict — auto-fix, report, blocking errors on severity="error" |
| **Landscape + audit** | `--check-guidelines` | Same as landscape | Read-only audit — report with warnings, no auto-fix, no blocking |

### Produced files

Artifacts are created under `<stem>_YYYYMMDD_HHMMSS/` in the current working directory (or in `--output`). Filenames with spaces or special characters are normalized automatically.

| File | Generated when |
|---|---|
| `<stem>.srt` | Always (sentence-merged in landscape, raw Whisper in `--short`) |
| `<stem>_corrected.srt` | ASR corrections, auto-splits or auto-merges were applied |
| `<stem>_report.txt` | ≥ 1 correction or ≥ 1 violation to report |
| `<stem>.<lang>.srt` or `<stem>_corrected.<lang>.srt` | Translation succeeded |

**Clean landscape pipeline:**
```
video_20260421_150309/
├── video.srt
└── video.en.srt
```

**Landscape with ASR corrections:**
```
video_20260421_150309/
├── video.srt
├── video_corrected.srt
├── video_report.txt
└── video_corrected.en.srt
```

**Shorts mode blocked by CPS error:**
```
video_20260421_150309/
├── video.srt
└── video_report.txt
```
Translation is skipped in `--short` mode when blocking errors (`severity="error"`) are detected. Landscape mode never blocks.

## Customizing branding (`branding.yaml`)

The `branding.yaml` file holds proper nouns and technical vocabulary to protect during ASR correction (Claude will not flag these terms as errors). The default file is bundled in the wheel. To customize, create a local file and point to it:

```bash
export BRANDING_YAML_PATH=/path/to/my_branding.yaml
```

**Schema** (validated by Pydantic — both keys are required, empty lists are accepted):

```yaml
noms_propres:            # proper nouns never to be corrected
  - Taverne Tech
vocabulaire_technique:   # technical terms never to be flagged as errors
  - Kubernetes
  - kubectl
```

A starter template lives at `branding.yaml.example` at the repo root. Copy it to `./branding.yaml` (which is gitignored) and point `BRANDING_YAML_PATH` at it.

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ANTHROPIC_API_KEY is missing` | Variable not set | Add to `.env` or export in shell |
| `unsupported extension` | Unsupported video format | Use `.mp4`, `.mkv`, `.mov` or `.avi` |
| `No subtitles detected` | Silent or inaudible video | Check that the video contains speech |
| Blocking errors (R5) | Severe YouTube guideline violations | Fix violations listed in the report |

**Exit codes:** `0` = success, `1` = error, `130` = interrupted (Ctrl+C).

## Development

Install dev dependencies:

```bash
uv sync --group dev
uv run pre-commit install   # enable git hooks (lint, format, typing)
```

Common commands:

```bash
# Tests
uv run pytest

# Lint + format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy subtitle_studio

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`) runs pre-commit and the test suite on Python 3.11 / 3.12 / 3.13 on every push and PR to `main`.

### YouTube upload (manual step)

After the pipeline, upload the subtitles manually in YouTube Studio:
1. Open the video in YouTube Studio
2. Go to **Subtitles**
3. Upload `<stem>_corrected.srt` (or `<stem>.srt`) for the source language
4. Upload `<stem>_corrected.en.srt` (or `<stem>.en.srt`) for the target language
