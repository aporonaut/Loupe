"""Tests for the analyzer protocol."""

import numpy as np

from loupe.analyzers.base import AnalyzerConfig, BaseAnalyzer, SharedModels
from loupe.core.models import AnalyzerResult, Tag


class MockAnalyzer:
    """A minimal analyzer satisfying the BaseAnalyzer protocol."""

    name = "mock"

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer=self.name,
            score=0.5,
            tags=[Tag(name="test_tag", confidence=0.9, category="composition")],
        )


class TestBaseAnalyzerProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        analyzer: BaseAnalyzer = MockAnalyzer()
        assert analyzer.name == "mock"

    def test_mock_analyze(self) -> None:
        analyzer = MockAnalyzer()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        config = AnalyzerConfig()
        shared = SharedModels()

        result = analyzer.analyze(image, config, shared)
        assert result.analyzer == "mock"
        assert result.score == 0.5
        assert len(result.tags) == 1


class TestAnalyzerConfig:
    def test_defaults(self) -> None:
        config = AnalyzerConfig()
        assert config.enabled is True
        assert config.confidence_threshold == 0.25
        assert config.params == {}

    def test_custom(self) -> None:
        config = AnalyzerConfig(
            enabled=False,
            confidence_threshold=0.5,
            params={"k_clusters": 6},
        )
        assert config.enabled is False
        assert config.params["k_clusters"] == 6
