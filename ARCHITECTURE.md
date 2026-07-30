# Architecture

Reference documentation for `subtitle_studio`'s internal design: package
layout, pipeline data flow, and implementation-level notes. For installation,
CLI usage, and end-user configuration (environment variables, CLI options,
output artifacts, `branding.yaml`), see `README.md`. For contribution rules
and conventions Claude Code must follow, see `CLAUDE.md`.

The pipeline is: **video file → audio extraction → transcription → verification → translation → SRT artifacts**

## Package layout

```
subtitle_studio/
├── cli.py          # Typer entry point — validation, per-stage Rich progress, --version, --verbose/--quiet, logging
├── pipeline.py     # Orchestrator — chains the 3 stages, writes artifacts
├── models.py       # PipelineConfigError, PipelineStepError
├── data/
│   └── branding.yaml   # Proper nouns and technical vocabulary (bundled in the wheel)
├── generate/
│   ├── audio.py        # ffmpeg audio extraction (16 kHz mono MP3 → tempfile)
│   ├── transcribe.py   # local/api dispatch; lazy import of stable_whisper
│   ├── transcribe_cloud.py   # OpenAI-compatible Whisper client (lazy import openai)
│   ├── sentence_merger.py # merge_into_sentences() — phrase-level fusion (YouTube-like landscape)
│   └── subtitle.py     # to_srt(), to_vtt(), wrap_text() (42 chars, 2 lines max)
├── detect/
│   ├── models.py       # Pydantic: Correction, GuidelineViolation, BrandingConfig + exceptions
│   ├── guidelines.py   # Deterministic YouTube checks (CPS, CPL, duration, gap, near-duplicate adjacent segments) — imported as `audit_guidelines` in pipeline.py to avoid shadowing the `check_guidelines` kwarg
│   ├── detector.py     # Claude API batching (50 segments/batch), prompt construction
│   ├── cps_autofix.py  # auto_fix_cps_violations() — Shorts mode only (split/downgrade)
│   ├── duration_autofix.py # auto_merge_short_segments() — Shorts mode only
│   └── srt_parser.py   # parse_srt() multi-encoding, write_srt() atomic, apply_corrections()
└── translate/
    └── translate.py    # translate_cues(), run_translation() — Claude API [N] format
```

## Data flow in pipeline.py

```
run_pipeline(shorts=False, check_guidelines=False)
  ├── _step_generate() → <stem>.srt
  │   ├── extract_audio()         # ffmpeg → tempfile
  │   ├── transcribe()            # stable_whisper (local) or transcribe_api (cloud)
  │   ├── to_subtitles()          # Whisper result → list[srt.Subtitle] with wrap_text
  │   └── merge_into_sentences()  # landscape mode only — YouTube-like ~5s / 84 chars segments
  │
  ├── gc.collect()                # ⚠️ INTENTIONAL — see the rule in CLAUDE.md
  │
  ├── _step_detect() → <stem>_corrected.srt (if modifications) or <stem>.srt
  │   ├── detect_errors()         # Claude API per batch — always runs
  │   ├── apply_corrections()     # always runs
  │   │
  │   ├── if shorts:              # Shorts mode — strict broadcast norms
  │   │     auto_merge_short_segments()
  │   │     auto_fix_cps_violations()
  │   │
  │   ├── if shorts or check_guidelines:
  │   │     audit_guidelines()    # read-only in landscape; enforced in Shorts (blocking errors)
  │   │
  │   └── _write_report()         # only if corrections/violations/fixes present
  │
  └── _step_translate() → <stem>.<lang>.srt (or <stem>_corrected.<lang>.srt)
      └── run_translation()       # Claude API [N] format
```

See README.md's "Pipeline and output artifacts" section for the full artifact
table and mode comparison from a user's perspective. A clean run producing
only `<stem>.srt` and `<stem>.<lang>.srt` is a **success** state, not an
error.

## Blocking error handling (R5)

Blocking errors apply **only in Shorts mode** (`shorts=True`). If the
verification stage detects `severity="error"` violations, the pipeline stops
before translation and raises `PipelineStepError`. Landscape mode never
blocks, even with `check_guidelines=True` — the audit is read-only and emits
warnings in the report.

## Landscape vs Shorts mode

| Aspect | Landscape (default) | Shorts (`--short`) |
|---|---|---|
| Segmentation | Sentence-level merge in `_step_generate` (~5 s / 84 chars, sentence-boundary preferred) | Raw Whisper segments |
| `auto_merge_short_segments` | skipped | applied |
| `auto_fix_cps_violations` | skipped | applied |
| `audit_guidelines` | skipped unless `check_guidelines=True` | applied |
| Blocking errors | never | on severity="error" |
| Report file | only if ≥ 1 ASR correction or (with `check_guidelines=True`) ≥ 1 violation | whenever violations/corrections/fixes exist |

### Duplicate-take detection (`near_duplicate`)

`check_near_duplicate_boundary()` in `detect/guidelines.py` flags adjacent
segments sharing a verbatim substring ≥ `DEFAULT_MIN_DUPLICATE_OVERLAP_CHARS`
(20 chars). It exists to catch a line spoken twice back-to-back in the source
footage (a re-recorded take left uncut) — Whisper transcribes both
occurrences correctly, so no hallucination filter or ASR correction removes
them, and the LLM coherence pass tends to leave paraphrased restatements
alone (its anti-hallucination rule favors an imperfect segment over an
invented fix). This is a `severity="warning"`, read-only signal only — never
auto-fixed, since choosing which take to keep is an editorial decision. Part
of `check_guidelines()`, so it follows the same gating as the rest of the
audit (Shorts always, landscape only with `check_guidelines=True`). Pairs
sharing the same segment `index` (CPS auto-split halves in Shorts mode,
before `write_srt` reindexes them) are skipped, so a split sentence never
gets reported as duplicating itself.

## Test layout

Tests are organized as a mirror of the package:

```
tests/
├── test_cli.py               # CLI entry point (typer.testing.CliRunner)
├── test_pipeline.py          # orchestration (stages mocked)
├── detect/
│   ├── conftest.py           # shared fixtures
│   ├── test_detector.py      # Claude detection (pure + mocked)
│   ├── test_guidelines.py    # deterministic checks (CPS, CPL, duration, gap, near-duplicate)
│   ├── test_cps_autofix.py   # Shorts-mode CPS auto-fix
│   ├── test_duration_autofix.py # Shorts-mode duration auto-merge
│   └── test_srt_parser.py    # SRT parsing
├── generate/
│   ├── test_subtitle.py      # SRT/VTT formatting
│   ├── test_sentence_merger.py # landscape sentence merging
│   └── test_transcribe.py    # hallucination filtering, backend dispatch
└── translate/
    └── test_translate.py     # Claude translation (mocked)
```

For the testing *conventions* (how to patch lazily-imported modules, which
path to mock for each pipeline stage), see `CLAUDE.md`.

## Performance

- **Default model: `large-v3`** (~3 GB, downloaded from HuggingFace Hub on first run to `~/.cache/huggingface/hub/`)
- On macOS, CTranslate2 runs **CPU-only** — no Metal/CoreML/Neural Engine acceleration
- `--device auto` and `--device cpu` are equivalent on macOS
- On an M4 Pro 24 GB, `large-v3` is recommended (best quality for French)

See `docs/solutions/performance-issues/whisper-model-selection-apple-silicon.md` for the full model selection guide.

## uv configuration

- The Python version is pinned via `.python-version` (created by `uv python pin 3.14`) — **not** in `pyproject.toml`
- Dev dependencies live under `[dependency-groups] dev` (PEP 735) — not in `[tool.uv] dev-dependencies` (deprecated)
- `uv sync --group dev` installs dev dependencies
- `uv tool install` resolves dependencies from PyPI regardless of `uv.lock` — for a reproducible environment, use `uv sync` and run via `uv run`

See `docs/solutions/packaging-issues/uv-pyproject-migration-gotchas.md` for the full reference.

## Requirement divergences

None. **R3** (per-stage progress) is implemented via a
`step_ctx: Callable[[str], AbstractContextManager[None]]` argument to
`run_pipeline`. The CLI provides a context manager that drives 3 Rich tasks
(`Extraction`, `Verification`, `Translation`) — dim before, cyan while running,
green ✓ once complete, red on failure. Callers that do not need progress
(tests, programmatic integrations) simply omit the argument and receive a
`nullcontext` by default.
