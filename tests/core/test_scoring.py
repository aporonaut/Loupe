"""Tests for the scoring module."""

import pytest

from loupe.core.models import AnalyzerResult
from loupe.core.scoring import compute_aggregate


def _result(name: str, score: float) -> AnalyzerResult:
    return AnalyzerResult(analyzer=name, score=score)


class TestComputeAggregate:
    def test_equal_weights(self) -> None:
        results = [_result("a", 0.8), _result("b", 0.6)]
        weights = {"a": 1.0, "b": 1.0}
        score, meta = compute_aggregate(results, weights)
        assert score == pytest.approx(0.7)
        assert meta.weights["a"] == pytest.approx(0.5)
        assert meta.weights["b"] == pytest.approx(0.5)

    def test_unequal_weights(self) -> None:
        results = [_result("a", 1.0), _result("b", 0.0)]
        weights = {"a": 3.0, "b": 1.0}
        score, meta = compute_aggregate(results, weights)
        assert score == pytest.approx(0.75)
        assert meta.weights["a"] == pytest.approx(0.75)
        assert meta.contributions["a"] == pytest.approx(0.75)
        assert meta.contributions["b"] == pytest.approx(0.0)

    def test_missing_dimension(self) -> None:
        """Weights shrink proportionally when a dimension is missing."""
        results = [_result("a", 0.8)]
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        score, meta = compute_aggregate(results, weights)
        # Only 'a' contributes, normalized to weight 1.0
        assert score == pytest.approx(0.8)
        assert "b" not in meta.weights

    def test_single_dimension_unreliable(self) -> None:
        results = [_result("a", 0.5)]
        weights = {"a": 1.0}
        score, meta = compute_aggregate(results, weights)
        assert score == pytest.approx(0.5)
        assert meta.reliable is False

    def test_two_dimensions_reliable(self) -> None:
        results = [_result("a", 0.5), _result("b", 0.5)]
        weights = {"a": 1.0, "b": 1.0}
        _, meta = compute_aggregate(results, weights)
        assert meta.reliable is True

    def test_empty_results(self) -> None:
        score, meta = compute_aggregate([], {"a": 1.0})
        assert score == 0.0
        assert meta.reliable is False

    def test_no_matching_weights(self) -> None:
        results = [_result("a", 0.8)]
        weights = {"b": 1.0}
        score, meta = compute_aggregate(results, weights)
        assert score == 0.0
        assert meta.reliable is False

    def test_contributions_sum_to_aggregate(self) -> None:
        results = [_result("a", 0.9), _result("b", 0.3), _result("c", 0.6)]
        weights = {"a": 2.0, "b": 1.0, "c": 1.0}
        score, meta = compute_aggregate(results, weights)
        total = sum(meta.contributions.values())
        assert score == pytest.approx(total)

    def test_all_six_dimensions(self) -> None:
        dims = ["composition", "color", "detail", "lighting", "subject", "style"]
        results = [_result(d, 0.5) for d in dims]
        weights = {d: 1.0 for d in dims}
        score, meta = compute_aggregate(results, weights)
        assert score == pytest.approx(0.5)
        assert len(meta.weights) == 6
        assert meta.reliable is True

    def test_preset_composition(self) -> None:
        """Composition preset should heavily favor composition score."""
        results = [_result("composition", 1.0), _result("color", 0.0)]
        weights = {"composition": 3.0, "color": 1.0}
        score, _ = compute_aggregate(results, weights)
        assert score == pytest.approx(0.75)
