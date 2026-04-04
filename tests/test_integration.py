"""End-to-end integration tests using real anime screenshots.

These tests require:
- Real images in tests/integration/ (gitignored)
- Downloaded ML models (run `loupe setup` first)
- GPU recommended but not required

Run with: uv run pytest tests/test_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from loupe.cli.main import app
from loupe.core.config import SCORING_PRESETS, load_config
from loupe.core.engine import Engine
from loupe.core.models import LoupeResult  # noqa: TC001 — used at runtime in fixtures
from loupe.io.sidecar import has_result, read_result

INTEGRATION_DIR = Path(__file__).parent / "integration"

runner = CliRunner()


def _has_images() -> bool:
    """Check whether the integration image directory exists and has images."""
    if not INTEGRATION_DIR.is_dir():
        return False
    return any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        for p in INTEGRATION_DIR.iterdir()
    )


def _image_paths() -> list[Path]:
    """Collect all image paths from the integration directory."""
    if not INTEGRATION_DIR.is_dir():
        return []
    return sorted(
        p
        for p in INTEGRATION_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


skip_no_images = pytest.mark.skipif(
    not _has_images(),
    reason="No integration test images in tests/integration/",
)


@pytest.fixture(scope="module")
def analyzed_results() -> list[LoupeResult]:
    """Run the full pipeline on all integration images (once per module).

    Returns the list of LoupeResult objects. Also writes sidecars
    so subsequent tests can verify sidecar I/O.
    """
    images = _image_paths()
    if not images:
        pytest.skip("No integration images available")

    cfg = load_config()
    engine = Engine(cfg)

    from loupe.analyzers.color import ColorAnalyzer
    from loupe.analyzers.composition import CompositionAnalyzer
    from loupe.analyzers.detail import DetailAnalyzer
    from loupe.analyzers.lighting import LightingAnalyzer
    from loupe.analyzers.style import StyleAnalyzer
    from loupe.analyzers.subject import SubjectAnalyzer

    engine.register_analyzer(ColorAnalyzer())
    engine.register_analyzer(CompositionAnalyzer())
    engine.register_analyzer(DetailAnalyzer())
    engine.register_analyzer(LightingAnalyzer())
    engine.register_analyzer(SubjectAnalyzer())
    engine.register_analyzer(StyleAnalyzer())

    results = engine.analyze_batch(images, force=True)
    assert len(results) > 0, "Expected at least one successful analysis result"
    return results


@skip_no_images
@pytest.mark.integration
class TestPipelineBasics:
    """Verify basic pipeline correctness on real images."""

    def test_all_images_produce_results(
        self, analyzed_results: list[LoupeResult]
    ) -> None:
        """Every image should produce a result (no silent failures)."""
        image_count = len(_image_paths())
        assert len(analyzed_results) == image_count

    def test_scores_in_valid_range(self, analyzed_results: list[LoupeResult]) -> None:
        """All scores (per-dimension and aggregate) must be in [0, 1]."""
        for result in analyzed_results:
            assert 0.0 <= result.aggregate_score <= 1.0, (
                f"{result.image_path.name}: aggregate "
                f"{result.aggregate_score} out of range"
            )
            for ar in result.analyzer_results:
                assert 0.0 <= ar.score <= 1.0, (
                    f"{result.image_path.name}/{ar.analyzer}: "
                    f"score {ar.score} out of range"
                )

    def test_all_six_dimensions_present(
        self, analyzed_results: list[LoupeResult]
    ) -> None:
        """Each result should have all 6 analyzer dimensions."""
        expected = {"composition", "color", "detail", "lighting", "subject", "style"}
        for result in analyzed_results:
            dimensions = {ar.analyzer for ar in result.analyzer_results}
            assert dimensions == expected, (
                f"{result.image_path.name}: missing {expected - dimensions}"
            )

    def test_tags_are_valid(self, analyzed_results: list[LoupeResult]) -> None:
        """All tags must have name, valid confidence, and correct category."""
        for result in analyzed_results:
            for ar in result.analyzer_results:
                for tag in ar.tags:
                    assert tag.name, f"Empty tag name in {ar.analyzer}"
                    assert 0.0 <= tag.confidence <= 1.0, (
                        f"Tag {tag.name} confidence {tag.confidence} out of range"
                    )
                    assert tag.category == ar.analyzer, (
                        f"Tag {tag.name} category {tag.category} != {ar.analyzer}"
                    )

    def test_aggregate_computed_correctly(
        self, analyzed_results: list[LoupeResult]
    ) -> None:
        """Aggregate score should match recomputation from dimension scores."""
        from loupe.core.scoring import compute_aggregate

        weights = SCORING_PRESETS["balanced"]
        for result in analyzed_results:
            expected_agg, _ = compute_aggregate(result.analyzer_results, weights)
            assert abs(result.aggregate_score - expected_agg) < 1e-4, (
                f"{result.image_path.name}: aggregate {result.aggregate_score} "
                f"!= recomputed {expected_agg}"
            )


@skip_no_images
@pytest.mark.integration
class TestSidecarIO:
    """Verify sidecar write/read roundtrip with real analysis results."""

    def test_sidecars_written(self, analyzed_results: list[LoupeResult]) -> None:
        """Each analyzed image should have a sidecar file."""
        for result in analyzed_results:
            assert has_result(result.image_path), (
                f"No sidecar for {result.image_path.name}"
            )

    def test_sidecar_roundtrip(self, analyzed_results: list[LoupeResult]) -> None:
        """Reading a sidecar back should produce an equivalent result."""
        for result in analyzed_results:
            loaded = read_result(result.image_path)
            assert loaded is not None, (
                f"Failed to read sidecar for {result.image_path.name}"
            )
            assert loaded.aggregate_score == result.aggregate_score
            assert len(loaded.analyzer_results) == len(result.analyzer_results)
            for orig, read in zip(
                result.analyzer_results, loaded.analyzer_results, strict=True
            ):
                assert orig.analyzer == read.analyzer
                assert abs(orig.score - read.score) < 1e-6


@skip_no_images
@pytest.mark.integration
class TestIncrementalAnalysis:
    """Verify skip/force behavior with existing sidecars."""

    def test_skip_already_analyzed(self, analyzed_results: list[LoupeResult]) -> None:
        """Re-analyzing without --force should skip (return None)."""
        cfg = load_config()
        engine = Engine(cfg)

        from loupe.analyzers.composition import CompositionAnalyzer

        engine.register_analyzer(CompositionAnalyzer())

        # Pick first image that was analyzed
        image_path = analyzed_results[0].image_path
        result = engine.analyze(image_path, force=False)
        assert result is None, "Expected skip for already-analyzed image"

    def test_force_reanalyze(self, analyzed_results: list[LoupeResult]) -> None:
        """Re-analyzing with --force should produce a fresh result."""
        cfg = load_config()
        engine = Engine(cfg)

        from loupe.analyzers.color import ColorAnalyzer
        from loupe.analyzers.composition import CompositionAnalyzer
        from loupe.analyzers.detail import DetailAnalyzer
        from loupe.analyzers.lighting import LightingAnalyzer
        from loupe.analyzers.style import StyleAnalyzer
        from loupe.analyzers.subject import SubjectAnalyzer

        engine.register_analyzer(ColorAnalyzer())
        engine.register_analyzer(CompositionAnalyzer())
        engine.register_analyzer(DetailAnalyzer())
        engine.register_analyzer(LightingAnalyzer())
        engine.register_analyzer(SubjectAnalyzer())
        engine.register_analyzer(StyleAnalyzer())

        image_path = analyzed_results[0].image_path
        result = engine.analyze(image_path, force=True)
        assert result is not None, "Expected result with --force"
        assert 0.0 <= result.aggregate_score <= 1.0


@skip_no_images
@pytest.mark.integration
class TestPresetRankingDifference:
    """Verify that different presets produce meaningfully different rankings."""

    def test_presets_change_rankings(self, analyzed_results: list[LoupeResult]) -> None:
        """'composition' and 'visual' presets should rank differently."""
        from loupe.core.scoring import compute_aggregate

        if len(analyzed_results) < 5:
            pytest.skip("Need at least 5 images for ranking comparison")

        def rank_order(preset_name: str) -> list[Path]:
            weights = SCORING_PRESETS[preset_name]
            scored = []
            for r in analyzed_results:
                agg, _ = compute_aggregate(r.analyzer_results, weights)
                scored.append((r.image_path, agg))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scored]

        comp_order = rank_order("composition")
        visual_order = rank_order("visual")

        # At least some images should swap positions
        differences = sum(
            1 for a, b in zip(comp_order, visual_order, strict=False) if a != b
        )
        assert differences > 0, (
            "Expected 'composition' and 'visual' presets to produce "
            "different rankings, but they were identical"
        )


@skip_no_images
@pytest.mark.integration
class TestCLIIntegration:
    """Verify CLI commands work end-to-end (after images are analyzed)."""

    def test_rank_command(self, analyzed_results: list[LoupeResult]) -> None:
        result = runner.invoke(app, ["rank", str(INTEGRATION_DIR)])
        assert result.exit_code == 0
        assert "Rankings" in result.output

    def test_report_command(self, analyzed_results: list[LoupeResult]) -> None:
        result = runner.invoke(app, ["report", str(INTEGRATION_DIR)])
        assert result.exit_code == 0
        assert "Report" in result.output

    def test_tags_command(self) -> None:
        result = runner.invoke(app, ["tags"])
        assert result.exit_code == 0
        assert "composition" in result.output
        assert "color" in result.output


@skip_no_images
@pytest.mark.integration
class TestScoreDistribution:
    """Sanity-check that score distributions aren't degenerate."""

    def test_scores_not_all_identical(
        self, analyzed_results: list[LoupeResult]
    ) -> None:
        """With diverse images, scores should vary."""
        if len(analyzed_results) < 10:
            pytest.skip("Need at least 10 images for distribution check")

        aggregates = [r.aggregate_score for r in analyzed_results]
        assert max(aggregates) - min(aggregates) > 0.05, (
            f"Score range too narrow: {min(aggregates):.3f}-{max(aggregates):.3f}"
        )

    def test_per_dimension_variance(self, analyzed_results: list[LoupeResult]) -> None:
        """Each dimension should show some variance across diverse images."""
        if len(analyzed_results) < 10:
            pytest.skip("Need at least 10 images for variance check")

        import statistics

        dimensions = {"composition", "color", "detail", "lighting", "subject", "style"}
        for dim in dimensions:
            scores = [
                ar.score
                for r in analyzed_results
                for ar in r.analyzer_results
                if ar.analyzer == dim
            ]
            if len(scores) < 2:
                continue
            std = statistics.stdev(scores)
            assert std > 0.01, (
                f"{dim}: standard deviation {std:.4f} is suspiciously low"
            )
