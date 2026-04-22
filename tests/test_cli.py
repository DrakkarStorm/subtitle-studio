"""Tests for the CLI entry point (subtitle_studio.cli)."""

from __future__ import annotations

import logging

from typer.testing import CliRunner

from subtitle_studio import __version__
from subtitle_studio.cli import _configure_logging, app

runner = CliRunner()

# CI runners default to 80 columns — Rich wraps the --help panel so aggressively
# that option flags can be split across lines or truncated. Force a wide terminal
# and disable colors to keep assertions deterministic.
_WIDE_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


class TestVersionFlag:
    def test_version_short(self) -> None:
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_version_long(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestHelp:
    def test_help_lists_version_flag(self) -> None:
        result = runner.invoke(app, ["--help"], env=_WIDE_ENV)
        assert result.exit_code == 0
        assert "--version" in result.stdout

    def test_help_lists_verbose_and_quiet(self) -> None:
        result = runner.invoke(app, ["--help"], env=_WIDE_ENV)
        assert result.exit_code == 0
        assert "--verbose" in result.stdout
        assert "--quiet" in result.stdout

    def test_help_lists_check_guidelines(self) -> None:
        result = runner.invoke(app, ["--help"], env=_WIDE_ENV)
        assert result.exit_code == 0
        assert "--check-guidelines" in result.stdout


class TestNoOpThresholdWarning:
    def test_warns_when_max_cps_passed_in_landscape(self, monkeypatch, tmp_path) -> None:
        """--max-cps without --short or --check-guidelines triggers a stderr warning."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")

        # The pipeline will fail (fake video), but the warning fires before that.
        result = runner.invoke(app, [str(video), "--max-cps", "25"])
        # Warning emitted to stderr regardless of exit code.
        assert "--max-cps" in result.stderr
        assert "ignored in landscape mode" in result.stderr

    def test_no_warning_when_max_cps_with_short(self, monkeypatch, tmp_path) -> None:
        """--max-cps with --short: no no-op warning."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")

        result = runner.invoke(app, [str(video), "--short", "--max-cps", "25"])
        assert "ignored in landscape mode" not in result.stderr

    def test_no_warning_when_max_cps_with_check_guidelines(self, monkeypatch, tmp_path) -> None:
        """--max-cps with --check-guidelines: no no-op warning."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")

        result = runner.invoke(app, [str(video), "--check-guidelines", "--max-cps", "25"])
        assert "ignored in landscape mode" not in result.stderr

    def test_no_warning_when_no_threshold_flags(self, monkeypatch, tmp_path) -> None:
        """Landscape without threshold flags: no warning."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")

        result = runner.invoke(app, [str(video)])
        assert "ignored in landscape mode" not in result.stderr


class TestConfigureLogging:
    def test_default_is_warning(self) -> None:
        _configure_logging(verbose=0, quiet=False)
        assert logging.getLogger().level == logging.WARNING

    def test_verbose_is_info(self) -> None:
        _configure_logging(verbose=1, quiet=False)
        assert logging.getLogger().level == logging.INFO

    def test_double_verbose_is_debug(self) -> None:
        _configure_logging(verbose=2, quiet=False)
        assert logging.getLogger().level == logging.DEBUG

    def test_quiet_is_error(self) -> None:
        _configure_logging(verbose=0, quiet=True)
        assert logging.getLogger().level == logging.ERROR

    def test_quiet_overrides_verbose(self) -> None:
        _configure_logging(verbose=2, quiet=True)
        assert logging.getLogger().level == logging.ERROR


class TestMissingApiKey:
    def test_exits_1_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["/tmp/nonexistent.mp4"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.stderr


class TestInvalidVideo:
    def test_exits_1_on_missing_file(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        result = runner.invoke(app, ["/tmp/does_not_exist_123456.mp4"])
        assert result.exit_code == 1
        assert "not found" in result.stderr

    def test_exits_1_on_bad_extension(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        bad = tmp_path / "video.xyz"
        bad.write_bytes(b"")
        result = runner.invoke(app, [str(bad)])
        assert result.exit_code == 1
        assert "extension" in result.stderr.lower()
