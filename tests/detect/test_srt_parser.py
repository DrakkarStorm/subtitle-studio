"""Tests for detect/srt_parser."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
import srt

from subtitle_studio.detect.models import Correction, SRTParseError
from subtitle_studio.detect.srt_parser import apply_corrections, parse_srt, write_srt


class TestParseSrt:
    def test_roundtrip(self, sample_srt_file: Path, sample_subtitles: list[srt.Subtitle]) -> None:
        loaded = parse_srt(sample_srt_file)
        assert len(loaded) == len(sample_subtitles)
        assert loaded[0].content == sample_subtitles[0].content
        assert loaded[1].content == sample_subtitles[1].content

    def test_bom_utf8_handled(self, tmp_path: Path) -> None:
        sub = srt.Subtitle(
            1,
            datetime.timedelta(seconds=0),
            datetime.timedelta(seconds=1),
            "Test BOM",
        )
        path = tmp_path / "bom.srt"
        path.write_bytes(b"\xef\xbb\xbf" + srt.compose([sub]).encode("utf-8"))
        result = parse_srt(path)
        assert result[0].content == "Test BOM"

    def test_latin1_fallback(self, tmp_path: Path) -> None:
        sub = srt.Subtitle(
            1,
            datetime.timedelta(seconds=0),
            datetime.timedelta(seconds=1),
            "Éàüç",
        )
        path = tmp_path / "latin1.srt"
        path.write_bytes(srt.compose([sub]).encode("latin-1"))
        result = parse_srt(path)
        assert result[0].content == "Éàüç"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SRTParseError):
            parse_srt(tmp_path / "nonexistent.srt")

    def test_invalid_srt_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.srt"
        path.write_text("this is not a valid SRT\n\n\n", encoding="utf-8")
        with pytest.raises(SRTParseError):
            parse_srt(path)


class TestWriteSrt:
    def test_write_and_reparse(self, sample_subtitles: list[srt.Subtitle], tmp_path: Path) -> None:
        path = tmp_path / "out.srt"
        write_srt(sample_subtitles, path)
        assert path.exists()
        reloaded = list(srt.parse(path.read_text(encoding="utf-8")))
        assert len(reloaded) == len(sample_subtitles)

    def test_atomic_write_no_tmp_residue(self, sample_subtitles: list[srt.Subtitle], tmp_path: Path) -> None:
        path = tmp_path / "out.srt"
        write_srt(sample_subtitles, path)
        tmp = path.with_suffix(".tmp")
        assert not tmp.exists()


class TestApplyCorrections:
    def test_applies_correction(self, sample_subtitles: list[srt.Subtitle], sample_correction: Correction) -> None:
        result = apply_corrections(sample_subtitles, [sample_correction])
        assert result[0].content == sample_correction.suggestion
        assert result[1].content == sample_subtitles[1].content

    def test_no_mutation(self, sample_subtitles: list[srt.Subtitle], sample_correction: Correction) -> None:
        original_content = sample_subtitles[0].content
        apply_corrections(sample_subtitles, [sample_correction])
        assert sample_subtitles[0].content == original_content

    def test_empty_corrections(self, sample_subtitles: list[srt.Subtitle]) -> None:
        result = apply_corrections(sample_subtitles, [])
        assert [s.content for s in result] == [s.content for s in sample_subtitles]

    def test_multiple_corrections(self, sample_subtitles: list[srt.Subtitle]) -> None:
        corrections = [
            Correction(segment=1, original="aller", suggestion="allé", raison="participe"),
            Correction(segment=3, original="c'est passer", suggestion="s'est passé", raison="homophone"),
        ]
        result = apply_corrections(sample_subtitles, corrections)
        assert result[0].content == "allé"
        assert result[2].content == "s'est passé"
