"""Tests for the engine orchestrator."""

from pathlib import Path

import numpy as np
from PIL import Image

from loupe.analyzers.base import AnalyzerConfig, SharedModels
from loupe.core.config import load_config
from loupe.core.engine import Engine
from loupe.core.models import AnalyzerResult, Tag
from loupe.io.sidecar import has_result


class StubAnalyzer:
    """A stub analyzer that returns a fixed score."""

    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self._score = score

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer=self.name,
            score=self._score,
            tags=[Tag(name=f"{self.name}_tag", confidence=0.9, category="composition")],
        )


def _make_image(tmp_path: Path, name: str = "test.png") -> Path:
    path = tmp_path / name
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(path)
    return path


class TestEngine:
    def test_zero_analyzers(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        image = _make_image(tmp_path)
        result = engine.analyze(image, force=True)
        assert result is not None
        assert result.aggregate_score == 0.0
        assert result.analyzer_results == []
        assert result.scoring.reliable is False

    def test_single_analyzer(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.8))
        image = _make_image(tmp_path)
        result = engine.analyze(image, force=True)
        assert result is not None
        assert result.aggregate_score > 0
        assert len(result.analyzer_results) == 1
        assert result.analyzer_results[0].analyzer == "composition"

    def test_two_analyzers_aggregate(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.8))
        engine.register_analyzer(StubAnalyzer("color", 0.6))
        image = _make_image(tmp_path)
        result = engine.analyze(image, force=True)
        assert result is not None
        assert 0.6 < result.aggregate_score < 0.8
        assert len(result.analyzer_results) == 2

    def test_incremental_skip(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.8))
        image = _make_image(tmp_path)

        # First run writes sidecar
        result1 = engine.analyze(image)
        assert result1 is not None
        assert has_result(image.resolve())

        # Second run skips
        result2 = engine.analyze(image)
        assert result2 is None

    def test_force_reanalyze(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.8))
        image = _make_image(tmp_path)

        engine.analyze(image, force=True)
        result = engine.analyze(image, force=True)
        assert result is not None

    def test_disabled_analyzer_skipped(self, tmp_path: Path) -> None:
        config = load_config()
        config.analyzers.color.enabled = False
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.8))
        engine.register_analyzer(StubAnalyzer("color", 0.6))
        image = _make_image(tmp_path)
        result = engine.analyze(image, force=True)
        assert result is not None
        assert len(result.analyzer_results) == 1
        assert result.analyzer_results[0].analyzer == "composition"

    def test_batch(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.5))
        images = [_make_image(tmp_path, f"img{i}.png") for i in range(3)]
        results = engine.analyze_batch(images, force=True)
        assert len(results) == 3

    def test_batch_progress_callback(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.5))
        images = [_make_image(tmp_path, f"img{i}.png") for i in range(2)]
        calls: list[tuple[int, int]] = []
        engine.analyze_batch(
            images, force=True, progress_callback=lambda i, t: calls.append((i, t))
        )
        assert calls[-1] == (2, 2)

    def test_unsupported_image_returns_none(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        gif = tmp_path / "test.gif"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(gif)
        result = engine.analyze(gif, force=True)
        assert result is None

    def test_writes_sidecar(self, tmp_path: Path) -> None:
        config = load_config()
        engine = Engine(config)
        engine.register_analyzer(StubAnalyzer("composition", 0.7))
        image = _make_image(tmp_path)
        engine.analyze(image, force=True)
        sidecar = image.parent / ".loupe" / f"{image.name}.json"
        assert sidecar.exists()
