"""Performance benchmarks for critical paths.

Scoring and classical analyzer benchmarks run without models.
Full-pipeline benchmarks require models + real images (marked integration).

Run scoring benchmarks:
    uv run pytest tests/test_benchmarks.py -m "not integration" --benchmark-only

Run all benchmarks (requires models + images in tests/integration/):
    uv run pytest tests/test_benchmarks.py --benchmark-only
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from loupe.analyzers.base import AnalyzerConfig, SharedModels

if TYPE_CHECKING:
    from loupe.core.engine import Engine
from loupe.core.models import AnalyzerResult
from loupe.core.scoring import compute_aggregate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INTEGRATION_DIR = Path(__file__).parent / "integration"


def _has_images() -> bool:
    if not INTEGRATION_DIR.is_dir():
        return False
    return any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        for p in INTEGRATION_DIR.iterdir()
    )


skip_no_images = pytest.mark.skipif(
    not _has_images(),
    reason="No integration test images in tests/integration/",
)


@pytest.fixture
def synthetic_scene() -> np.ndarray:
    """A 1920x1080 synthetic scene with some structure for benchmarking."""
    rng = np.random.default_rng(42)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Background gradient
    img[:, :, 0] = np.linspace(40, 100, 1920, dtype=np.uint8)
    img[:, :, 1] = np.linspace(60, 120, 1920, dtype=np.uint8)
    img[:, :, 2] = np.linspace(80, 140, 1920, dtype=np.uint8)
    # Bright subject region
    img[300:700, 600:1000] = rng.integers(180, 255, (400, 400, 3), dtype=np.uint8)
    # Some edge structure
    img[200, :] = 255
    img[:, 960] = 255
    return img


@pytest.fixture
def empty_shared() -> SharedModels:
    """Empty SharedModels for classical analyzers."""
    return SharedModels()


@pytest.fixture
def mock_shared(synthetic_scene: np.ndarray) -> SharedModels:
    """SharedModels with a synthetic segmentation mask."""
    h, w = synthetic_scene.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    # Subject region
    mask[300:700, 600:1000] = 1.0
    return SharedModels(segmentation_mask=mask)


@pytest.fixture
def default_config() -> AnalyzerConfig:
    return AnalyzerConfig()


# ---------------------------------------------------------------------------
# Scoring benchmarks (no model dependencies)
# ---------------------------------------------------------------------------


class TestScoringBenchmarks:
    """Benchmark aggregate scoring computation."""

    def test_compute_aggregate_6_dims(self, benchmark: pytest.BenchmarkFixture) -> None:  # type: ignore[name-defined]
        """Benchmark scoring with all 6 dimensions."""
        results = [
            AnalyzerResult(analyzer="composition", score=0.65),
            AnalyzerResult(analyzer="color", score=0.72),
            AnalyzerResult(analyzer="detail", score=0.58),
            AnalyzerResult(analyzer="lighting", score=0.71),
            AnalyzerResult(analyzer="subject", score=0.45),
            AnalyzerResult(analyzer="style", score=0.53),
        ]
        weights = {
            "composition": 1.0,
            "color": 1.0,
            "detail": 1.0,
            "lighting": 1.0,
            "subject": 1.0,
            "style": 0.5,
        }

        score, meta = benchmark(compute_aggregate, results, weights)
        assert 0.0 <= score <= 1.0
        assert meta.reliable


# ---------------------------------------------------------------------------
# Classical analyzer benchmarks (no model dependencies)
# ---------------------------------------------------------------------------


class TestClassicalAnalyzerBenchmarks:
    """Benchmark analyzers that don't require ML models."""

    def test_composition_analyzer(
        self,
        benchmark: pytest.BenchmarkFixture,  # type: ignore[name-defined]
        synthetic_scene: np.ndarray,
        empty_shared: SharedModels,
        default_config: AnalyzerConfig,
    ) -> None:
        """Benchmark composition analyzer on a 1080p image."""
        from loupe.analyzers.composition import CompositionAnalyzer

        analyzer = CompositionAnalyzer()
        result = benchmark(
            analyzer.analyze, synthetic_scene, default_config, empty_shared
        )
        assert 0.0 <= result.score <= 1.0

    def test_color_analyzer(
        self,
        benchmark: pytest.BenchmarkFixture,  # type: ignore[name-defined]
        synthetic_scene: np.ndarray,
        empty_shared: SharedModels,
        default_config: AnalyzerConfig,
    ) -> None:
        """Benchmark color analyzer on a 1080p image."""
        from loupe.analyzers.color import ColorAnalyzer

        analyzer = ColorAnalyzer()
        result = benchmark(
            analyzer.analyze, synthetic_scene, default_config, empty_shared
        )
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Model-dependent analyzer benchmarks (require shared model outputs)
# ---------------------------------------------------------------------------


class TestModelAnalyzerBenchmarks:
    """Benchmark analyzers that consume shared model outputs.

    These use a synthetic segmentation mask rather than running the
    actual models, so they measure analyzer logic time only.
    """

    def test_detail_analyzer(
        self,
        benchmark: pytest.BenchmarkFixture,  # type: ignore[name-defined]
        synthetic_scene: np.ndarray,
        mock_shared: SharedModels,
        default_config: AnalyzerConfig,
    ) -> None:
        """Benchmark detail analyzer with synthetic segmentation mask."""
        from loupe.analyzers.detail import DetailAnalyzer

        analyzer = DetailAnalyzer()
        result = benchmark(
            analyzer.analyze, synthetic_scene, default_config, mock_shared
        )
        assert 0.0 <= result.score <= 1.0

    def test_lighting_analyzer(
        self,
        benchmark: pytest.BenchmarkFixture,  # type: ignore[name-defined]
        synthetic_scene: np.ndarray,
        mock_shared: SharedModels,
        default_config: AnalyzerConfig,
    ) -> None:
        """Benchmark lighting analyzer with synthetic segmentation mask."""
        from loupe.analyzers.lighting import LightingAnalyzer

        analyzer = LightingAnalyzer()
        result = benchmark(
            analyzer.analyze, synthetic_scene, default_config, mock_shared
        )
        assert 0.0 <= result.score <= 1.0

    def test_subject_analyzer(
        self,
        benchmark: pytest.BenchmarkFixture,  # type: ignore[name-defined]
        synthetic_scene: np.ndarray,
        mock_shared: SharedModels,
        default_config: AnalyzerConfig,
    ) -> None:
        """Benchmark subject analyzer with synthetic segmentation mask."""
        from loupe.analyzers.subject import SubjectAnalyzer

        analyzer = SubjectAnalyzer()
        result = benchmark(
            analyzer.analyze, synthetic_scene, default_config, mock_shared
        )
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Full pipeline benchmarks (require models + real images)
# ---------------------------------------------------------------------------


@skip_no_images
@pytest.mark.integration
class TestPipelineBenchmarks:
    """Benchmark the full analysis pipeline on real images.

    These tests load actual models and process real images.
    Use ``--benchmark-min-rounds=3`` for stable measurements.
    """

    @pytest.fixture(scope="class")
    def engine(self) -> Engine:
        """Create and configure an engine with all models loaded."""
        from loupe.analyzers.color import ColorAnalyzer
        from loupe.analyzers.composition import CompositionAnalyzer
        from loupe.analyzers.detail import DetailAnalyzer
        from loupe.analyzers.lighting import LightingAnalyzer
        from loupe.analyzers.style import StyleAnalyzer
        from loupe.analyzers.subject import SubjectAnalyzer
        from loupe.core.config import load_config
        from loupe.core.engine import Engine

        cfg = load_config()
        eng = Engine(cfg)
        eng.register_analyzer(ColorAnalyzer())
        eng.register_analyzer(CompositionAnalyzer())
        eng.register_analyzer(DetailAnalyzer())
        eng.register_analyzer(LightingAnalyzer())
        eng.register_analyzer(SubjectAnalyzer())
        eng.register_analyzer(StyleAnalyzer())
        eng.ensure_models_loaded()
        return eng

    @pytest.fixture
    def sample_image(self) -> Path:
        """Pick the first available integration image."""
        images = sorted(
            p
            for p in INTEGRATION_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        return images[0]

    def test_single_image_analysis(
        self,
        benchmark: pytest.BenchmarkFixture,  # type: ignore[name-defined]
        engine: Engine,
        sample_image: Path,
    ) -> None:
        """Benchmark full pipeline for a single image."""
        result = benchmark(engine.analyze, sample_image, force=True)
        assert result is not None
        assert 0.0 <= result.aggregate_score <= 1.0
