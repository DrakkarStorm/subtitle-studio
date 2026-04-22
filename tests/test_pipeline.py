"""Tests for the orchestration pipeline."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import srt

from subtitle_studio.detect.models import CpsAutoFix, DurationAutoFix
from subtitle_studio.models import PipelineConfigError, PipelineStepError
from subtitle_studio.pipeline import _make_output_dir, _step_detect, run_pipeline


@pytest.fixture
def sample_srt_path(tmp_path: Path) -> Path:
    """Temporary SRT file with 2 valid subtitles."""
    subtitles = [
        srt.Subtitle(
            index=1,
            start=datetime.timedelta(seconds=1),
            end=datetime.timedelta(seconds=3),
            content="Bonjour le monde.",
        ),
        srt.Subtitle(
            index=2,
            start=datetime.timedelta(seconds=4),
            end=datetime.timedelta(seconds=6),
            content="Comment ça va ?",
        ),
    ]
    path = tmp_path / "test.srt"
    path.write_text(srt.compose(subtitles), encoding="utf-8")
    return path


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    """Fake video file (empty content, valid extension)."""
    path = tmp_path / "test.mp4"
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------------------
# Tests _make_output_dir
# ---------------------------------------------------------------------------


class TestMakeOutputDir:
    def test_returns_path_with_stem_and_timestamp(self, fake_video: Path) -> None:
        result = _make_output_dir(fake_video)
        assert result.name.startswith("test_")
        assert result.parent == Path.cwd()

    def test_special_chars_sanitized(self, tmp_path: Path) -> None:
        video = tmp_path / "my video (2024).mp4"
        video.write_bytes(b"")
        result = _make_output_dir(video)
        # Special characters replaced with _
        assert "(" not in result.name
        assert ")" not in result.name

    def test_collision_adds_suffix(self, fake_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """On collision, a numeric suffix is appended."""
        fixed_ts = "20260101_120000"

        with patch("subtitle_studio.pipeline.datetime") as mock_dt:
            mock_dt.datetime.now.return_value.strftime.return_value = fixed_ts
            mock_dt.date = datetime.date

            # Create the first directory to force the collision
            first = Path.cwd() / f"test_{fixed_ts}"
            first.mkdir(exist_ok=True)
            try:
                result = _make_output_dir(fake_video)
                assert result.name == f"test_{fixed_ts}_1"
            finally:
                first.rmdir()


# ---------------------------------------------------------------------------
# run_pipeline tests (stages mocked)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_returns_output_dir(self, fake_video: Path, sample_srt_path: Path, tmp_path: Path) -> None:
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            result = run_pipeline(fake_video, output_dir=out_dir)

        assert result == out_dir

    def test_step_ctx_invoked_in_order(self, fake_video: Path, sample_srt_path: Path, tmp_path: Path) -> None:
        """step_ctx must be called once per stage, in order."""
        from contextlib import contextmanager

        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        calls: list[str] = []

        @contextmanager
        def recorder(name: str):
            calls.append(f"start:{name}")
            try:
                yield
            finally:
                calls.append(f"end:{name}")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, step_ctx=recorder)

        assert calls == [
            "start:Extraction",
            "end:Extraction",
            "start:Verification",
            "end:Verification",
            "start:Translation",
            "end:Translation",
        ]

    def test_step_ctx_exits_on_failure(self, fake_video: Path, tmp_path: Path) -> None:
        """The context manager must receive the exception from the failing stage."""
        from contextlib import contextmanager

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        seen: list[tuple[str, bool]] = []

        @contextmanager
        def recorder(name: str):
            failed = False
            try:
                yield
            except BaseException:
                failed = True
                raise
            finally:
                seen.append((name, failed))

        with (
            patch(
                "subtitle_studio.pipeline._step_generate",
                side_effect=PipelineStepError("Extraction", "boom"),
            ),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineStepError),
        ):
            run_pipeline(fake_video, output_dir=out_dir, step_ctx=recorder)

        assert seen == [("Extraction", True)]

    def test_step_generate_error_raises_pipeline_step_error(self, fake_video: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch(
                "subtitle_studio.pipeline._step_generate",
                side_effect=PipelineStepError("Extraction", "ffmpeg absent"),
            ),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineStepError, match="Extraction"),
        ):
            run_pipeline(fake_video, output_dir=out_dir)

    def test_invalid_branding_raises_pipeline_config_error(self, fake_video: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch(
                "subtitle_studio.pipeline.load_branding",
                side_effect=ValueError("invalid branding"),
            ),
            pytest.raises(PipelineConfigError, match="invalid branding"),
        ):
            run_pipeline(fake_video, output_dir=out_dir)

    def test_gc_collect_called_after_step_generate(
        self, fake_video: Path, sample_srt_path: Path, tmp_path: Path
    ) -> None:
        """gc.collect() must be called after stage 1 to free the Whisper model."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect") as mock_gc,
        ):
            run_pipeline(fake_video, output_dir=out_dir)

        mock_gc.assert_called_once()

    def test_shorts_true_passes_max_chars_32_to_steps(
        self, fake_video: Path, sample_srt_path: Path, tmp_path: Path
    ) -> None:
        """shorts=True must pass max_chars=32 to _step_generate and _step_translate."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path) as mock_gen,
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path) as mock_det,
            patch("subtitle_studio.pipeline._step_translate", return_value=translated) as mock_trans,
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, shorts=True)

        assert mock_gen.call_args.kwargs["max_chars"] == 32
        assert mock_det.call_args.kwargs["max_cpl"] == 32
        assert mock_det.call_args.kwargs["warn_cpl"] == 28
        assert mock_trans.call_args.kwargs["max_chars"] == 32

    def test_shorts_false_uses_default_max_chars(self, fake_video: Path, sample_srt_path: Path, tmp_path: Path) -> None:
        """shorts=False (default) must pass max_chars=42 (MAX_CHARS) to the stages."""
        from subtitle_studio.generate.subtitle import MAX_CHARS

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path) as mock_gen,
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir)

        assert mock_gen.call_args.kwargs["max_chars"] == MAX_CHARS

    def test_max_cps_forwarded_to_step_detect(self, fake_video: Path, sample_srt_path: Path, tmp_path: Path) -> None:
        """run_pipeline(max_cps=25.0) forwards max_cps=25.0 to _step_detect."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path) as mock_det,
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, max_cps=25.0)

        assert mock_det.call_args.kwargs["max_cps"] == 25.0

    def test_min_gap_forwarded_to_step_detect(self, fake_video: Path, sample_srt_path: Path, tmp_path: Path) -> None:
        """run_pipeline(min_gap=40) forwards min_gap_ms=40 to _step_detect."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path) as mock_det,
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, min_gap=40)

        assert mock_det.call_args.kwargs["min_gap_ms"] == 40

    def test_explicit_max_cpl_wins_over_shorts(self, fake_video: Path, sample_srt_path: Path, tmp_path: Path) -> None:
        """run_pipeline(shorts=True, max_cpl=38) → _step_detect receives max_cpl=38, not 32."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path) as mock_det,
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, shorts=True, max_cpl=38)

        assert mock_det.call_args.kwargs["max_cpl"] == 38

    def test_warn_cps_gte_max_cps_raises_config_error(self, fake_video: Path, tmp_path: Path) -> None:
        """warn_cps >= max_cps raises PipelineConfigError before any extraction."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineConfigError, match="warn-cps"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, warn_cps=22.0, max_cps=20.0)

    def test_warn_cpl_gte_max_cpl_raises_config_error(self, fake_video: Path, tmp_path: Path) -> None:
        """warn_cpl >= max_cpl raises PipelineConfigError before any extraction."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineConfigError, match="warn-cpl"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, warn_cpl=45, max_cpl=40)

    def test_defaults_unchanged_when_no_threshold_flags(
        self, fake_video: Path, sample_srt_path: Path, tmp_path: Path
    ) -> None:
        """Without threshold flags, _step_detect receives the default constants."""
        from subtitle_studio.detect.guidelines import DEFAULT_MAX_CPS, DEFAULT_MIN_GAP_MS

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path) as mock_det,
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir)

        assert mock_det.call_args.kwargs["max_cps"] == DEFAULT_MAX_CPS
        assert mock_det.call_args.kwargs["min_gap_ms"] == DEFAULT_MIN_GAP_MS

    def test_target_lang_forwarded_to_step_translate(
        self, fake_video: Path, sample_srt_path: Path, tmp_path: Path
    ) -> None:
        """run_pipeline(target_lang='pt') forwards target_lang='pt' to _step_translate."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        translated = tmp_path / "test.pt.srt"
        translated.write_text("", encoding="utf-8")

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_translate", return_value=translated) as mock_trans,
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, target_lang="pt")

        assert mock_trans.call_args.kwargs["target_lang"] == "pt"

    def test_unsupported_target_lang_raises_config_error(self, fake_video: Path, tmp_path: Path) -> None:
        """Unknown target_lang raises PipelineConfigError before any extraction."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineConfigError, match="target_lang"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, target_lang="zh")

    def test_zero_min_gap_raises_config_error(self, fake_video: Path, tmp_path: Path) -> None:
        """min_gap=0 raises PipelineConfigError before any extraction."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineConfigError, match="min-gap"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, min_gap=0)

    def test_negative_max_cps_raises_config_error(self, fake_video: Path, tmp_path: Path) -> None:
        """max_cps=-1.0 raises PipelineConfigError before any extraction."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineConfigError, match="max-cps"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, max_cps=-1.0, warn_cps=-2.0)

    def test_warn_cps_equal_max_cps_raises_config_error(self, fake_video: Path, tmp_path: Path) -> None:
        """warn_cps == max_cps raises PipelineConfigError (check is >=, not just >)."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineConfigError, match="warn-cps"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, warn_cps=18.0, max_cps=18.0)

    def test_config_error_does_not_create_output_dir(self, fake_video: Path, tmp_path: Path) -> None:
        """PipelineConfigError raised before mkdir — no orphan directory created."""
        out_dir = tmp_path / "orphan_output"
        # out_dir n'existe pas encore

        with (
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            pytest.raises(PipelineConfigError),
        ):
            run_pipeline(fake_video, output_dir=out_dir, target_lang="xx")

        assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Tests _step_detect interne
# ---------------------------------------------------------------------------


class TestStepDetect:
    def test_passes_max_chars_to_auto_fix(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """_step_detect forwards max_chars to auto_fix_cps_violations."""
        from unittest.mock import MagicMock

        client = MagicMock()
        branding = MagicMock()

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=([], []),
            ) as mock_autofix,
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[]),
        ):
            _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                max_chars=32,
                shorts=True,
            )

        assert mock_autofix.call_args.kwargs["max_chars"] == 32

    def test_cps_error_downgraded_unblocks_pipeline(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """CPS error violation on a downgraded segment → no PipelineStepError."""
        from unittest.mock import MagicMock

        from subtitle_studio.detect.models import GuidelineViolation

        client = MagicMock()
        branding = MagicMock()

        cps_fix = CpsAutoFix(segment=1, action="downgrade", original_cps=25.0, description="test")
        cps_violation = GuidelineViolation(segment=1, rule="cps", severity="error", description="25 car/s")

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=(list(srt.parse(sample_srt_path.read_text())), [cps_fix]),
            ),
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[cps_violation]),
        ):
            # Must not raise PipelineStepError
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=True,
            )

        assert result == sample_srt_path  # no corrected.srt (downgrade only, no corrections)

    def test_corrected_srt_written_when_splits_exist(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """CPS splits → corrected.srt written even without ASR corrections."""
        from unittest.mock import MagicMock

        import srt as srt_lib

        client = MagicMock()
        branding = MagicMock()

        original_subs = list(srt_lib.parse(sample_srt_path.read_text()))
        cps_fix = CpsAutoFix(segment=1, action="split", original_cps=25.0, description="test")
        # Simulated sub-segments (renumbered indices)
        split_subs = original_subs + [
            srt_lib.Subtitle(
                index=3,
                start=datetime.timedelta(seconds=1),
                end=datetime.timedelta(seconds=2),
                content="Extra segment.",
            )
        ]

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=(split_subs, [cps_fix]),
            ),
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[]),
        ):
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=True,
            )

        corrected = tmp_path / "test_corrected.srt"
        assert result == corrected
        assert corrected.exists()

    def test_corrected_srt_not_written_for_downgrade_only(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """Downgrades only (no splits, no ASR corrections) → no corrected.srt."""
        from unittest.mock import MagicMock

        import srt as srt_lib

        client = MagicMock()
        branding = MagicMock()

        original_subs = list(srt_lib.parse(sample_srt_path.read_text()))
        cps_fix = CpsAutoFix(segment=1, action="downgrade", original_cps=25.0, description="test")

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=(original_subs, [cps_fix]),
            ),
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[]),
        ):
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
            )

        corrected = tmp_path / "test_corrected.srt"
        assert result == sample_srt_path
        assert not corrected.exists()


# ---------------------------------------------------------------------------
# Integration tests — auto_merge_short_segments in _step_detect
# ---------------------------------------------------------------------------


class TestStepDetectDurationMerge:
    def test_duration_merge_passed_max_chars(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """_step_detect forwards max_chars to auto_merge_short_segments."""
        from unittest.mock import MagicMock

        client = MagicMock()
        branding = MagicMock()

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_merge_short_segments",
                return_value=([], []),
            ) as mock_merge,
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=([], []),
            ),
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[]),
        ):
            _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                max_chars=32,
                shorts=True,
            )

        assert mock_merge.call_args.kwargs["max_chars"] == 32

    def test_corrected_srt_written_when_merges(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """Non-empty duration_fixes → corrected.srt written."""
        from unittest.mock import MagicMock

        import srt as srt_lib

        client = MagicMock()
        branding = MagicMock()

        original_subs = list(srt_lib.parse(sample_srt_path.read_text()))
        duration_fix = DurationAutoFix(segment=2, merged_with=1, original_duration_s=0.3, direction="backward")

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_merge_short_segments",
                return_value=(original_subs, [duration_fix]),
            ),
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=(original_subs, []),
            ),
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[]),
        ):
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=True,
            )

        corrected = tmp_path / "test_corrected.srt"
        assert result == corrected
        assert corrected.exists()

    def test_no_corrected_srt_when_no_merges(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """No merge, no correction, no split → no corrected.srt."""
        from unittest.mock import MagicMock

        import srt as srt_lib

        client = MagicMock()
        branding = MagicMock()

        original_subs = list(srt_lib.parse(sample_srt_path.read_text()))

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_merge_short_segments",
                return_value=(original_subs, []),
            ),
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=(original_subs, []),
            ),
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[]),
        ):
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=True,
            )

        corrected = tmp_path / "test_corrected.srt"
        assert result == sample_srt_path
        assert not corrected.exists()

    def test_duration_section_in_report(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """Report contains the 'Duration auto-merge' section when duration_fixes is non-empty."""
        from unittest.mock import MagicMock

        import srt as srt_lib

        client = MagicMock()
        branding = MagicMock()

        original_subs = list(srt_lib.parse(sample_srt_path.read_text()))
        duration_fix = DurationAutoFix(segment=2, merged_with=1, original_duration_s=0.3, direction="backward")

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch(
                "subtitle_studio.pipeline.auto_merge_short_segments",
                return_value=(original_subs, [duration_fix]),
            ),
            patch(
                "subtitle_studio.pipeline.auto_fix_cps_violations",
                return_value=(original_subs, []),
            ),
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[]),
        ):
            _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=True,
            )

        report = tmp_path / "test_report.txt"
        assert report.exists()
        content = report.read_text(encoding="utf-8")
        assert "Duration auto-merge" in content


# ---------------------------------------------------------------------------
# Landscape mode behavior — shorts=False path
# ---------------------------------------------------------------------------


class TestStepDetectLandscape:
    def test_landscape_skips_auto_fix_and_auto_merge(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """In landscape mode, auto_fix_cps_violations and auto_merge_short_segments must not run."""
        from unittest.mock import MagicMock

        client = MagicMock()
        branding = MagicMock()

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch("subtitle_studio.pipeline.auto_fix_cps_violations") as mock_autofix,
            patch("subtitle_studio.pipeline.auto_merge_short_segments") as mock_merge,
            patch("subtitle_studio.pipeline.audit_guidelines") as mock_audit,
        ):
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=False,
                check_guidelines=False,
            )

        mock_autofix.assert_not_called()
        mock_merge.assert_not_called()
        mock_audit.assert_not_called()
        assert result == sample_srt_path
        assert not (tmp_path / "test_report.txt").exists()
        assert not (tmp_path / "test_corrected.srt").exists()

    def test_landscape_with_corrections_writes_corrected_srt(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """Landscape + ASR corrections → writes corrected.srt and a report with only corrections."""
        from unittest.mock import MagicMock

        from subtitle_studio.detect.models import Correction

        client = MagicMock()
        branding = MagicMock()
        correction = Correction(segment=1, original="Bonjour", suggestion="Bonjour,", raison="ponctuation")

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[correction]),
        ):
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=False,
                check_guidelines=False,
            )

        corrected = tmp_path / "test_corrected.srt"
        report = tmp_path / "test_report.txt"
        assert result == corrected
        assert corrected.exists()
        assert report.exists()
        report_text = report.read_text()
        assert "Correction report" in report_text
        assert "YouTube compliance" not in report_text  # no guideline section

    def test_run_pipeline_forwards_check_guidelines_to_step_detect(
        self, fake_video: Path, sample_srt_path: Path, tmp_path: Path
    ) -> None:
        """run_pipeline(check_guidelines=True) must pass the flag down to _step_detect."""
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path) as mock_detect,
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, check_guidelines=True)

        assert mock_detect.call_args.kwargs["check_guidelines"] is True
        assert mock_detect.call_args.kwargs["shorts"] is False

    def test_run_pipeline_forwards_shorts_to_step_generate(
        self, fake_video: Path, sample_srt_path: Path, tmp_path: Path
    ) -> None:
        """run_pipeline(shorts=True) must pass shorts=True to _step_generate
        (so the merger is skipped upstream of detection)."""
        translated = tmp_path / "test.en.srt"
        translated.write_text("", encoding="utf-8")
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            patch("subtitle_studio.pipeline._step_generate", return_value=sample_srt_path) as mock_gen,
            patch("subtitle_studio.pipeline._step_detect", return_value=sample_srt_path),
            patch("subtitle_studio.pipeline._step_translate", return_value=translated),
            patch("subtitle_studio.pipeline.anthropic.Anthropic"),
            patch("subtitle_studio.pipeline.load_branding"),
            patch("subtitle_studio.pipeline.gc.collect"),
        ):
            run_pipeline(fake_video, output_dir=out_dir, shorts=True)

        assert mock_gen.call_args.kwargs["shorts"] is True

    def test_step_generate_skips_merger_when_shorts_true(self, fake_video: Path, tmp_path: Path) -> None:
        """_step_generate(shorts=True) must not call merge_into_sentences."""
        from unittest.mock import MagicMock

        from subtitle_studio.pipeline import _step_generate

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        fake_segment = MagicMock(start=0.0, end=2.0, text="hello world")
        fake_result = MagicMock(segments=[fake_segment])

        with (
            patch("subtitle_studio.pipeline.extract_audio", return_value=tmp_path / "audio.mp3"),
            patch("subtitle_studio.pipeline.transcribe", return_value=fake_result),
            patch("subtitle_studio.pipeline.merge_into_sentences") as mock_merge,
        ):
            (tmp_path / "audio.mp3").write_bytes(b"")
            _step_generate(fake_video, out_dir, backend="local", shorts=True)

        mock_merge.assert_not_called()

    def test_step_generate_calls_merger_when_shorts_false(self, fake_video: Path, tmp_path: Path) -> None:
        """_step_generate(shorts=False) must call merge_into_sentences."""
        from unittest.mock import MagicMock

        from subtitle_studio.pipeline import _step_generate

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        fake_segment = MagicMock(start=0.0, end=2.0, text="hello world")
        fake_result = MagicMock(segments=[fake_segment])

        with (
            patch("subtitle_studio.pipeline.extract_audio", return_value=tmp_path / "audio.mp3"),
            patch("subtitle_studio.pipeline.transcribe", return_value=fake_result),
            patch(
                "subtitle_studio.pipeline.merge_into_sentences",
                side_effect=lambda subs: subs,
            ) as mock_merge,
        ):
            (tmp_path / "audio.mp3").write_bytes(b"")
            _step_generate(fake_video, out_dir, backend="local", shorts=False)

        mock_merge.assert_called_once()

    def test_landscape_with_check_guidelines_runs_audit_only(self, sample_srt_path: Path, tmp_path: Path) -> None:
        """Landscape + check_guidelines=True → audit runs, no auto-fix, no blocking on errors."""
        from unittest.mock import MagicMock

        from subtitle_studio.detect.models import GuidelineViolation

        client = MagicMock()
        branding = MagicMock()
        violation = GuidelineViolation(segment=1, rule="cps", severity="error", description="too fast")

        with (
            patch("subtitle_studio.pipeline.detect_errors", return_value=[]),
            patch("subtitle_studio.pipeline.auto_fix_cps_violations") as mock_autofix,
            patch("subtitle_studio.pipeline.auto_merge_short_segments") as mock_merge,
            patch("subtitle_studio.pipeline.audit_guidelines", return_value=[violation]),
        ):
            # Even with a severity=error violation, landscape must not raise.
            result = _step_detect(
                sample_srt_path,
                tmp_path,
                client,
                "",
                branding,
                "model",
                50,
                shorts=False,
                check_guidelines=True,
            )

        mock_autofix.assert_not_called()
        mock_merge.assert_not_called()
        assert result == sample_srt_path
        report = tmp_path / "test_report.txt"
        assert report.exists()
        assert "YouTube compliance" in report.read_text()
