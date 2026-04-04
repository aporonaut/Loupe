"""Tests for sidecar I/O."""

from pathlib import Path

from loupe.core.models import LoupeResult
from loupe.io.sidecar import has_result, read_result, write_result


class TestSidecarIO:
    def test_write_creates_sidecar_dir(self, sample_loupe_result: LoupeResult) -> None:
        sidecar_path = write_result(sample_loupe_result)
        assert sidecar_path.exists()
        assert sidecar_path.parent.name == ".loupe"

    def test_has_result_true_after_write(
        self, sample_loupe_result: LoupeResult
    ) -> None:
        write_result(sample_loupe_result)
        assert has_result(sample_loupe_result.image_path) is True

    def test_has_result_false_before_write(self, tmp_path: Path) -> None:
        assert has_result(tmp_path / "nonexistent.png") is False

    def test_read_roundtrip(self, sample_loupe_result: LoupeResult) -> None:
        write_result(sample_loupe_result)
        restored = read_result(sample_loupe_result.image_path)
        assert restored is not None
        assert restored.aggregate_score == sample_loupe_result.aggregate_score
        assert len(restored.analyzer_results) == 2
        assert restored.analyzer_results[0].analyzer == "composition"
        assert restored.analyzer_results[0].score == 0.8
        assert restored.analyzer_results[0].tags[0].name == "rule_of_thirds"

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_result(tmp_path / "missing.png") is None

    def test_read_corrupted_returns_none(self, tmp_path: Path) -> None:
        image_path = tmp_path / "corrupt.png"
        sidecar_dir = tmp_path / ".loupe"
        sidecar_dir.mkdir()
        (sidecar_dir / "corrupt.png.json").write_text("not valid json")
        assert read_result(image_path) is None
