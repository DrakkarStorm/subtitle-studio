"""Sequential orchestration of the subtitle-studio pipeline."""

from __future__ import annotations

import datetime
import gc
import re
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import anthropic
import srt

from .detect.cps_autofix import auto_fix_cps_violations
from .detect.detector import BATCH_SIZE, detect_errors, load_branding
from .detect.detector import MODEL as DETECT_MODEL
from .detect.duration_autofix import auto_merge_short_segments
from .detect.guidelines import (
    DEFAULT_MAX_CPL,
    DEFAULT_MAX_CPS,
    DEFAULT_MIN_GAP_MS,
    DEFAULT_WARN_CPL,
    DEFAULT_WARN_CPS,
    SHORTS_MAX_CPL,
    SHORTS_WARN_CPL,
)
from .detect.guidelines import check_guidelines as audit_guidelines
from .detect.models import BrandingConfig, ClaudeAPIError, Correction, CpsAutoFix, DurationAutoFix, GuidelineViolation
from .detect.srt_parser import apply_corrections, parse_srt, write_srt
from .generate.audio import extract_audio
from .generate.sentence_merger import merge_into_sentences
from .generate.subtitle import MAX_CHARS, to_subtitles
from .generate.transcribe import transcribe
from .models import PipelineConfigError, PipelineStepError
from .translate.translate import SUPPORTED_LANGS as TRANSLATE_SUPPORTED_LANGS
from .translate.translate import run_translation

# Re-export exceptions so cli.py can import them from .pipeline
__all__ = ["run_pipeline", "PipelineStepError", "PipelineConfigError"]

# Package directory — used to resolve the bundled branding.yaml
_PACKAGE_DIR = Path(__file__).parent


def run_pipeline(
    video_path: Path,
    *,
    output_dir: Path | None = None,
    context: str = "",
    backend: str = "local",
    branding_path: Path | None = None,
    model: str = DETECT_MODEL,
    batch_size: int = BATCH_SIZE,
    shorts: bool = False,
    target_lang: str = "en",
    max_cps: float | None = None,
    warn_cps: float | None = None,
    max_cpl: int | None = None,
    warn_cpl: int | None = None,
    min_gap: int | None = None,
    check_guidelines: bool = False,
    step_ctx: Callable[[str], AbstractContextManager[None]] | None = None,
) -> Path:
    """Orchestrate the 3 stages. Returns the output directory.

    Args:
        model: Claude model for both detection and translation. For stage-level
            overrides, call ``_step_detect`` and ``_step_translate`` directly
            with a custom ``model`` argument.
        shorts: ``True`` enables strict Shorts formatting — YouTube guideline
            enforcement (CPS/CPL/duration/gap), auto-fix, auto-merge, blocking
            errors. ``False`` (default) produces YouTube-like long segments via
            sentence merging and skips all guideline enforcement.
        check_guidelines: Opt-in read-only guideline audit in landscape mode.
            No-op when ``shorts=True`` (shorts always audits). When enabled in
            landscape, emits violations in the report without auto-fixing or
            blocking.
        step_ctx: Context manager factory invoked with the stage name
            (`"Extraction"`, `"Verification"`, `"Translation"`). Lets the
            caller (the CLI) display a per-stage progress bar.
            Default: `nullcontext` (no output).

    Raises:
        PipelineStepError: When a stage fails unrecoverably.
        PipelineConfigError: On initial validation (invalid branding, etc.).
    """
    out = output_dir or _make_output_dir(video_path)

    # SHORTS_MAX_CPL also drives the wrap width (wrap_text max_chars) — same value.
    # CPS, duration and gap are not adjusted in Shorts mode (intentional — see brainstorm).
    max_chars = SHORTS_MAX_CPL if shorts else MAX_CHARS

    # Threshold resolution: explicit flag > --short preset > default constant
    eff_max_cpl = max_cpl if max_cpl is not None else (SHORTS_MAX_CPL if shorts else DEFAULT_MAX_CPL)
    eff_warn_cpl = warn_cpl if warn_cpl is not None else (SHORTS_WARN_CPL if shorts else DEFAULT_WARN_CPL)
    eff_max_cps = max_cps if max_cps is not None else DEFAULT_MAX_CPS
    eff_warn_cps = warn_cps if warn_cps is not None else DEFAULT_WARN_CPS
    eff_min_gap = min_gap if min_gap is not None else DEFAULT_MIN_GAP_MS

    if eff_warn_cps >= eff_max_cps:
        raise PipelineConfigError(f"--warn-cps ({eff_warn_cps}) must be < --max-cps ({eff_max_cps})")
    if eff_warn_cpl >= eff_max_cpl:
        raise PipelineConfigError(f"--warn-cpl ({eff_warn_cpl}) must be < --max-cpl ({eff_max_cpl})")
    if target_lang not in TRANSLATE_SUPPORTED_LANGS:
        raise PipelineConfigError(
            f"target_lang must be one of: {', '.join(TRANSLATE_SUPPORTED_LANGS.keys())} (got: {target_lang!r})"
        )
    for _name, _val in [
        ("max-cps", eff_max_cps),
        ("warn-cps", eff_warn_cps),
        ("max-cpl", float(eff_max_cpl)),
        ("warn-cpl", float(eff_warn_cpl)),
        ("min-gap", float(eff_min_gap)),
    ]:
        if _val <= 0:
            raise PipelineConfigError(f"--{_name} must be > 0 (got: {_val})")

    out.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(max_retries=3)
    resolved_branding_path = branding_path or _default_branding_path()

    try:
        branding = load_branding(resolved_branding_path)
    except ValueError as exc:
        raise PipelineConfigError(str(exc)) from exc

    _ctx = step_ctx or (lambda _name: nullcontext())

    # Stage 1 — Extraction
    with _ctx("Extraction"):
        srt_path = _step_generate(video_path, out, backend, max_chars=max_chars, shorts=shorts)

    # Free the Whisper model (~3 GB) before the API calls
    gc.collect()

    # Stage 2 — Verification
    with _ctx("Verification"):
        srt_for_translation = _step_detect(
            srt_path,
            out,
            client,
            context,
            branding,
            model,
            batch_size,
            max_cpl=eff_max_cpl,
            warn_cpl=eff_warn_cpl,
            max_chars=max_chars,
            max_cps=eff_max_cps,
            warn_cps=eff_warn_cps,
            min_gap_ms=eff_min_gap,
            shorts=shorts,
            check_guidelines=check_guidelines,
        )

    # Stage 3 — Translation
    with _ctx("Translation"):
        _translated = _step_translate(
            srt_for_translation,
            out,
            client,
            model,
            max_chars=max_chars,
            target_lang=target_lang,
        )

    return out


def _step_generate(
    video_path: Path,
    out: Path,
    backend: str,
    max_chars: int = MAX_CHARS,
    shorts: bool = False,
) -> Path:
    audio_path: Path | None = None
    try:
        audio_path = extract_audio(video_path)
        result = transcribe(audio_path, backend=backend)
        subtitles = to_subtitles(result, max_chars)
        if not subtitles:
            raise PipelineStepError("Extraction", "No subtitles detected — silent video?")

        # Landscape mode produces YouTube-like long phrase-based segments.
        # Shorts mode keeps the raw Whisper segmentation so the downstream
        # auto-fix/auto-merge stages can apply strict broadcast norms.
        if not shorts:
            subtitles = merge_into_sentences(subtitles)

        srt_path = out / (video_path.stem + ".srt")
        write_srt(subtitles, srt_path)

        return srt_path
    except PipelineStepError:
        raise
    except Exception as exc:
        raise PipelineStepError("Extraction", str(exc)) from exc
    finally:
        if audio_path is not None and audio_path.exists():
            audio_path.unlink()


def _step_detect(
    srt_path: Path,
    out: Path,
    client: anthropic.Anthropic,
    context: str,
    branding: BrandingConfig,
    model: str,
    batch_size: int,
    max_cpl: int = DEFAULT_MAX_CPL,
    warn_cpl: int = DEFAULT_WARN_CPL,
    max_chars: int = MAX_CHARS,
    max_cps: float = DEFAULT_MAX_CPS,
    warn_cps: float = DEFAULT_WARN_CPS,
    min_gap_ms: int = DEFAULT_MIN_GAP_MS,
    shorts: bool = False,
    check_guidelines: bool = False,
) -> Path:
    try:
        subtitles = parse_srt(srt_path)

        # 1. ASR corrections (Claude) — always runs
        corrections = detect_errors(
            subtitles,
            context,
            branding,
            client,
            batch_size=batch_size,
            model=model,
        )
        working_subs = apply_corrections(subtitles, corrections) if corrections else subtitles

        # Auto-fixes (duration merge + CPS split) only apply in Shorts mode.
        # Landscape mode keeps the sentence-merged segmentation as-is.
        duration_fixes: list[DurationAutoFix] = []
        cps_fixes: list[CpsAutoFix] = []
        if shorts:
            working_subs, duration_fixes = auto_merge_short_segments(working_subs, max_chars=max_chars)
            working_subs, cps_fixes = auto_fix_cps_violations(working_subs, max_cps=max_cps, max_chars=max_chars)

        # Guideline audit: Shorts enforces; landscape opts in via check_guidelines.
        violations: list[GuidelineViolation] = []
        if shorts or check_guidelines:
            violations = audit_guidelines(
                working_subs,
                max_cps=max_cps,
                warn_cps=warn_cps,
                max_cpl=max_cpl,
                warn_cpl=warn_cpl,
                min_gap_ms=min_gap_ms,
            )
            if shorts:
                # Downgrade CPS violations on irreducible segments: error → warning
                downgraded = {f.segment for f in cps_fixes if f.action == "downgrade"}
                violations = _apply_downgrades(violations, downgraded)

        # Report (skipped when there is nothing to report)
        _write_report(violations, corrections, duration_fixes, cps_fixes, subtitles, working_subs, out, srt_path.stem)

        # R5 — blocking errors only apply in Shorts mode.
        if shorts:
            blocking = [v for v in violations if v.severity == "error"]
            if blocking:
                raise PipelineStepError(
                    "Verification",
                    f"{len(blocking)} blocking error(s) detected.\n"
                    + "\n".join(f"  • {v.rule} (segment {v.segment})" for v in blocking),
                )

        # Write the corrected SRT if any modification was applied
        has_splits = any(f.action == "split" for f in cps_fixes)
        if corrections or has_splits or duration_fixes:
            srt_corrected = out / (srt_path.stem + "_corrected.srt")
            write_srt(working_subs, srt_corrected)
            return srt_corrected

        return srt_path
    except PipelineStepError:
        raise
    except ClaudeAPIError as exc:
        raise PipelineStepError("Verification", str(exc)) from exc
    except Exception as exc:
        raise PipelineStepError("Verification", str(exc)) from exc


def _step_translate(
    srt_input: Path,
    out: Path,
    client: anthropic.Anthropic,
    model: str,
    max_chars: int = MAX_CHARS,
    target_lang: str = "en",
) -> Path:
    try:
        return run_translation(srt_input, out, client, model=model, max_chars=max_chars, target_lang=target_lang)
    except ClaudeAPIError as exc:
        raise PipelineStepError("Translation", str(exc)) from exc
    except Exception as exc:
        raise PipelineStepError("Translation", str(exc)) from exc


def _apply_downgrades(
    violations: list[GuidelineViolation],
    downgraded_segments: set[int],
) -> list[GuidelineViolation]:
    """Convert CPS error violations to warnings for downgraded segments."""
    if not downgraded_segments:
        return violations
    result: list[GuidelineViolation] = []
    for v in violations:
        if v.rule == "cps" and v.severity == "error" and v.segment in downgraded_segments:
            result.append(
                GuidelineViolation(
                    segment=v.segment,
                    rule=v.rule,
                    severity="warning",
                    description=v.description + " [downgraded — irreducible]",
                )
            )
        else:
            result.append(v)
    return result


def _make_output_dir(video_path: Path) -> Path:
    safe_stem = re.sub(r"[^\w\-. ]", "_", video_path.stem)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = Path.cwd() / f"{safe_stem}_{timestamp}"
    if candidate.exists():
        i = 1
        while (c := Path.cwd() / f"{safe_stem}_{timestamp}_{i}").exists():
            i += 1
        candidate = c
    return candidate


def _default_branding_path() -> Path:
    """Resolve the bundled branding.yaml, or the path from BRANDING_YAML_PATH."""
    import os

    if env := os.environ.get("BRANDING_YAML_PATH"):
        return Path(env)
    return _PACKAGE_DIR / "data" / "branding.yaml"


# ---------------------------------------------------------------------------
# Text report helpers (ported from detect_wrong_subtitle/cli.py)
# ---------------------------------------------------------------------------


def _fmt_timedelta(td: datetime.timedelta) -> str:
    """Format a timedelta in SRT form: HH:MM:SS,mmm"""
    total_ms = int(td.total_seconds() * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


_RULE_LABELS: dict[str, str] = {
    "cps": "Reading speed (CPS)",
    "cpl": "Line length (CPL)",
    "lines": "Number of lines",
    "duration_min": "Minimum duration",
    "duration_max": "Maximum duration",
    "gap": "Gap between subtitles",
}


def _write_report(
    violations: list[GuidelineViolation],
    corrections: list[Correction],
    duration_fixes: list[DurationAutoFix],
    cps_fixes: list[CpsAutoFix],
    original_subs: list[srt.Subtitle],
    working_subs: list[srt.Subtitle],
    out_dir: Path,
    stem: str,
) -> None:
    """Write the text report if violations, corrections or auto-fixes exist."""
    if not violations and not corrections and not duration_fixes and not cps_fixes:
        return

    # Index by segment number for fast timecode lookup
    original_by_index = {sub.index: sub for sub in original_subs}
    working_by_index = {sub.index: sub for sub in working_subs}

    date_str = datetime.date.today().isoformat()
    sections: list[list[str]] = []

    if violations:
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        block: list[str] = [
            f"=== YouTube compliance — {stem}.srt ===",
            (
                f"Generated on {date_str} | {len(violations)} violation(s)"
                f" — {len(errors)} error(s), {len(warnings)} warning(s)"
            ),
            "",
        ]
        for v in violations:
            sub = working_by_index.get(v.segment)
            timecode = f"[{_fmt_timedelta(sub.start)} --> {_fmt_timedelta(sub.end)}]" if sub is not None else ""
            label = "[ERROR] " if v.severity == "error" else "[WARN]  "
            rule_label = _RULE_LABELS.get(v.rule, v.rule)
            block += [
                f"{label} Segment {v.segment} — {timecode}",
                f"Rule    : {rule_label}",
                f"Detail  : {v.description}",
                "",
            ]
        sections.append(block)

    if corrections:
        block = [
            f"=== Correction report — {stem}.srt ===",
            f"Generated on {date_str} | {len(corrections)} error(s) detected over {len(original_subs)} segments",
            "",
        ]
        for c in corrections:
            sub = original_by_index.get(c.segment)
            if sub is None:
                continue
            timecode = f"[{_fmt_timedelta(sub.start)} --> {_fmt_timedelta(sub.end)}]"
            block += [
                timecode,
                f"Before  : {c.original}",
                f"After   : {c.suggestion}",
                f"Reason  : {c.raison}",
                "",
            ]
        sections.append(block)

    if duration_fixes:
        block = [
            f"=== Duration auto-merge — {stem}.srt ===",
            f"Generated on {date_str} | {len(duration_fixes)} segment(s) merged",
            "",
        ]
        for df in duration_fixes:
            sub = original_by_index.get(df.segment)
            timecode = f"[{_fmt_timedelta(sub.start)} --> {_fmt_timedelta(sub.end)}]" if sub is not None else ""
            direction_label = "previous segment" if df.segment > df.merged_with else "next segment"
            block += [
                f"Segment {df.segment} — {timecode}",
                f"Action  : MERGED with {direction_label} (idx {df.merged_with})",
                f"Orig dur: {df.original_duration_s:.2f}s",
                "",
            ]
        sections.append(block)

    if cps_fixes:
        splits = [cf for cf in cps_fixes if cf.action == "split"]
        downgrades = [cf for cf in cps_fixes if cf.action == "downgrade"]
        block = [
            f"=== CPS auto-fix — {stem}.srt ===",
            (
                f"Generated on {date_str} | {len(cps_fixes)} segment(s) processed"
                f" — {len(splits)} split, {len(downgrades)} downgraded"
            ),
            "",
        ]
        for cf in cps_fixes:
            sub = original_by_index.get(cf.segment)
            timecode = f"[{_fmt_timedelta(sub.start)} --> {_fmt_timedelta(sub.end)}]" if sub is not None else ""
            action_label = "SPLIT" if cf.action == "split" else "DOWNGRADED (irreducible)"
            block += [
                f"Segment {cf.segment} — {timecode}",
                f"Action  : {action_label}",
                f"Orig CPS: {cf.original_cps:.1f} chars/s",
                "",
            ]
        sections.append(block)

    report_path = out_dir / (stem + "_report.txt")
    report_path.write_text(
        "\n".join("\n".join(section) for section in sections),
        encoding="utf-8",
    )
