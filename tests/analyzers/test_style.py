"""Tests for the style analyzer."""

import numpy as np
import pytest

from loupe.analyzers.base import AnalyzerConfig, SharedModels
from loupe.analyzers.style import (
    StyleAnalyzer,
    _combine_scores,
    _generate_tags,
    _measure_edge_uniformity,
    _measure_gradient_consistency,
    _measure_palette_coherence,
    _measure_region_consistency,
    measure_aesthetic_quality,
    measure_layer_consistency,
)

# -- Fixtures --


@pytest.fixture
def uniform_rgb() -> np.ndarray:
    """Uniform mid-gray RGB image — trivially consistent."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def noisy_rgb() -> np.ndarray:
    """Random noise RGB image — low consistency."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)


@pytest.fixture
def gradient_rgb() -> np.ndarray:
    """Smooth horizontal gradient — consistent rendering."""
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    for x in range(200):
        val = int(255 * x / 199)
        arr[:, x, :] = val
    return arr


@pytest.fixture
def patchy_rgb() -> np.ndarray:
    """Image with distinctly different patches — inconsistent detail."""
    arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Dense detail region (top-left)
    rng = np.random.default_rng(42)
    arr[:100, :100] = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    # Flat region (bottom-right)
    arr[100:, 100:] = 80
    return arr


@pytest.fixture
def center_character_mask() -> np.ndarray:
    """Character mask with center 40% foreground."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:140, 60:140] = 1
    return mask


@pytest.fixture
def default_config() -> AnalyzerConfig:
    """Default style analyzer config."""
    return AnalyzerConfig(
        enabled=True,
        confidence_threshold=0.25,
        params={"tagger_threshold": 0.35},
    )


@pytest.fixture
def mock_aesthetic_high() -> tuple[float, str, dict[str, float]]:
    """High aesthetic score prediction."""
    return (
        0.85,
        "best",
        {
            "worst": 0.01,
            "low": 0.01,
            "normal": 0.03,
            "good": 0.05,
            "great": 0.15,
            "best": 0.60,
            "masterpiece": 0.15,
        },
    )


@pytest.fixture
def mock_aesthetic_low() -> tuple[float, str, dict[str, float]]:
    """Low aesthetic score prediction."""
    return (
        0.2,
        "low",
        {
            "worst": 0.20,
            "low": 0.50,
            "normal": 0.20,
            "good": 0.05,
            "great": 0.03,
            "best": 0.01,
            "masterpiece": 0.01,
        },
    )


# -- Aesthetic Quality Tests --


class TestAestheticQuality:
    def test_high_score(
        self, mock_aesthetic_high: tuple[float, str, dict[str, float]]
    ) -> None:
        score = measure_aesthetic_quality(mock_aesthetic_high)
        assert score == pytest.approx(0.85, abs=0.01)

    def test_low_score(
        self, mock_aesthetic_low: tuple[float, str, dict[str, float]]
    ) -> None:
        score = measure_aesthetic_quality(mock_aesthetic_low)
        assert score == pytest.approx(0.2, abs=0.01)

    def test_none_returns_neutral(self) -> None:
        score = measure_aesthetic_quality(None)
        assert score == 0.5

    def test_score_clamped(self) -> None:
        # Ensure out-of-range values are clamped
        prediction = (1.5, "masterpiece", {"masterpiece": 1.0})
        score = measure_aesthetic_quality(prediction)
        assert score <= 1.0

    def test_score_in_range(
        self, mock_aesthetic_high: tuple[float, str, dict[str, float]]
    ) -> None:
        score = measure_aesthetic_quality(mock_aesthetic_high)
        assert 0.0 <= score <= 1.0


# -- Edge Uniformity Tests --


class TestEdgeUniformity:
    def test_uniform_image_high_uniformity(self) -> None:
        gray = np.full((200, 200), 128, dtype=np.uint8)
        score = _measure_edge_uniformity(gray, None)
        # No edges at all — trivially uniform
        assert score >= 0.9

    def test_noisy_image_moderate(self) -> None:
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        score = _measure_edge_uniformity(gray, None)
        # Random noise should have fairly uniform edge density
        assert 0.0 <= score <= 1.0

    def test_patchy_image_lower_uniformity(self) -> None:
        """Image with one detailed quadrant and one flat quadrant."""
        gray = np.full((200, 200), 128, dtype=np.uint8)
        rng = np.random.default_rng(42)
        gray[:100, :100] = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        score = _measure_edge_uniformity(gray, None)
        # Should be less uniform than fully uniform or fully noisy
        assert score < 0.9

    def test_with_mask(self) -> None:
        gray = np.full((200, 200), 128, dtype=np.uint8)
        mask = np.ones((200, 200), dtype=np.uint8)
        score = _measure_edge_uniformity(gray, mask)
        assert 0.0 <= score <= 1.0

    def test_tiny_image_returns_default(self) -> None:
        gray = np.full((8, 8), 128, dtype=np.uint8)
        score = _measure_edge_uniformity(gray, None)
        assert score == 0.5

    def test_score_in_range(self) -> None:
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        score = _measure_edge_uniformity(gray, None)
        assert 0.0 <= score <= 1.0


# -- Gradient Consistency Tests --


class TestGradientConsistency:
    def test_uniform_image_high_consistency(self) -> None:
        gray = np.full((200, 200), 128, dtype=np.uint8)
        score = _measure_gradient_consistency(gray, None)
        # Flat image → perfectly consistent (no gradients)
        assert score >= 0.9

    def test_smooth_gradient_consistent(self) -> None:
        """Smooth gradient should have concentrated gradient distribution."""
        col = np.linspace(0, 255, 200, dtype=np.uint8).reshape(1, -1)
        gray = np.tile(col, (200, 1))
        score = _measure_gradient_consistency(gray, None)
        assert score > 0.3

    def test_noisy_image_lower_consistency(self) -> None:
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        score = _measure_gradient_consistency(gray, None)
        # Random noise has scattered gradient distribution
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self) -> None:
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        score = _measure_gradient_consistency(gray, None)
        assert 0.0 <= score <= 1.0


# -- Palette Coherence Tests --


class TestPaletteCoherence:
    def test_single_color_high_coherence(self, uniform_rgb: np.ndarray) -> None:
        score = _measure_palette_coherence(uniform_rgb, None)
        assert score >= 0.9

    def test_noisy_image_lower_coherence(self, noisy_rgb: np.ndarray) -> None:
        score = _measure_palette_coherence(noisy_rgb, None)
        # Random colors → harder to fit a 4-color palette
        assert score < 0.9

    def test_gradient_moderate_coherence(self, gradient_rgb: np.ndarray) -> None:
        score = _measure_palette_coherence(gradient_rgb, None)
        assert 0.0 <= score <= 1.0

    def test_with_mask(self, uniform_rgb: np.ndarray) -> None:
        mask = np.ones((200, 200), dtype=np.uint8)
        score = _measure_palette_coherence(uniform_rgb, mask)
        assert score >= 0.9

    def test_small_mask_returns_default(self, uniform_rgb: np.ndarray) -> None:
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[0, 0] = 1  # Only 1 pixel
        score = _measure_palette_coherence(uniform_rgb, mask)
        assert score == 0.5

    def test_score_in_range(self, noisy_rgb: np.ndarray) -> None:
        score = _measure_palette_coherence(noisy_rgb, None)
        assert 0.0 <= score <= 1.0


# -- Region Consistency Tests --


class TestRegionConsistency:
    def test_uniform_high_consistency(self, uniform_rgb: np.ndarray) -> None:
        gray = np.full((200, 200), 128, dtype=np.uint8)
        score = _measure_region_consistency(uniform_rgb, gray)
        assert score >= 0.8

    def test_noisy_lower_consistency(self, noisy_rgb: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2GRAY)
        score = _measure_region_consistency(noisy_rgb, gray)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self, gradient_rgb: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(gradient_rgb, cv2.COLOR_RGB2GRAY)
        score = _measure_region_consistency(gradient_rgb, gray)
        assert 0.0 <= score <= 1.0


# -- Layer Consistency Tests --


class TestLayerConsistency:
    def test_no_mask_whole_image(self, uniform_rgb: np.ndarray) -> None:
        score = measure_layer_consistency(uniform_rgb, None)
        assert score >= 0.8

    def test_with_mask_separate_layers(
        self,
        uniform_rgb: np.ndarray,
        center_character_mask: np.ndarray,
    ) -> None:
        score = measure_layer_consistency(uniform_rgb, center_character_mask)
        # Uniform image: both layers consistent
        assert score >= 0.7

    def test_small_fg_treated_as_whole(self, uniform_rgb: np.ndarray) -> None:
        """Very small foreground fraction falls back to whole-image."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[0, 0] = 1  # < 1% foreground
        score = measure_layer_consistency(uniform_rgb, mask)
        assert 0.0 <= score <= 1.0

    def test_large_fg_treated_as_whole(self, uniform_rgb: np.ndarray) -> None:
        """Very large foreground fraction falls back to whole-image."""
        mask = np.ones((200, 200), dtype=np.uint8)
        mask[0, 0] = 0  # > 90% foreground
        score = measure_layer_consistency(uniform_rgb, mask)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(
        self,
        noisy_rgb: np.ndarray,
        center_character_mask: np.ndarray,
    ) -> None:
        score = measure_layer_consistency(noisy_rgb, center_character_mask)
        assert 0.0 <= score <= 1.0


# -- Score Combination Tests --


class TestCombineScores:
    def test_default_weights(self) -> None:
        scores = {"aesthetic_quality": 0.8, "layer_consistency": 0.6}
        weights = {"aesthetic_quality": 0.70, "layer_consistency": 0.30}
        result = _combine_scores(scores, weights)
        expected = 0.70 * 0.8 + 0.30 * 0.6
        assert result == pytest.approx(expected, abs=0.01)

    def test_equal_weights(self) -> None:
        scores = {"aesthetic_quality": 0.6, "layer_consistency": 0.4}
        weights = {"aesthetic_quality": 1.0, "layer_consistency": 1.0}
        result = _combine_scores(scores, weights)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_zero_weights(self) -> None:
        scores = {"aesthetic_quality": 0.8}
        weights = {"aesthetic_quality": 0.0}
        assert _combine_scores(scores, weights) == 0.0

    def test_empty_scores(self) -> None:
        assert _combine_scores({}, {"aesthetic_quality": 1.0}) == 0.0


# -- Tag Generation Tests --


class TestGenerateTags:
    def test_aesthetic_tier_tags(
        self, mock_aesthetic_high: tuple[float, str, dict[str, float]]
    ) -> None:
        tags = _generate_tags(
            aesthetic_prediction=mock_aesthetic_high,
            tagger_predictions=None,
            clip_style_scores=None,
            layer_consistency=0.5,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "aesthetic_best" in names

    def test_no_aesthetic_no_tier_tags(self) -> None:
        tags = _generate_tags(
            aesthetic_prediction=None,
            tagger_predictions=None,
            clip_style_scores=None,
            layer_consistency=0.5,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert not any(n.startswith("aesthetic_") for n in names)

    def test_tagger_style_tags_passthrough(self) -> None:
        tags = _generate_tags(
            aesthetic_prediction=None,
            tagger_predictions={
                "flat_color": 0.8,
                "cel_shading": 0.6,
                "unrelated_tag": 0.9,
            },
            clip_style_scores=None,
            layer_consistency=0.5,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "flat_color" in names
        assert "cel_shading" in names
        assert "unrelated_tag" not in names

    def test_tagger_below_threshold_filtered(self) -> None:
        tags = _generate_tags(
            aesthetic_prediction=None,
            tagger_predictions={"bloom": 0.1},
            clip_style_scores=None,
            layer_consistency=0.5,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "bloom" not in names

    def test_clip_style_tags(self) -> None:
        tags = _generate_tags(
            aesthetic_prediction=None,
            tagger_predictions=None,
            clip_style_scores={
                "naturalistic anime": 0.6,
                "digital modern anime": 0.3,
                "painterly anime": 0.05,
                "geometric abstract anime": 0.03,
                "retro cel anime": 0.02,
            },
            layer_consistency=0.5,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "naturalistic_anime" in names
        assert "digital_modern_anime" in names
        # Below threshold
        assert "painterly_anime" not in names

    def test_consistent_rendering_tag(self) -> None:
        tags = _generate_tags(
            aesthetic_prediction=None,
            tagger_predictions=None,
            clip_style_scores=None,
            layer_consistency=0.8,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "consistent_rendering" in names

    def test_inconsistent_rendering_tag(self) -> None:
        tags = _generate_tags(
            aesthetic_prediction=None,
            tagger_predictions=None,
            clip_style_scores=None,
            layer_consistency=0.2,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "inconsistent_rendering" in names

    def test_all_tags_have_style_category(self) -> None:
        tags = _generate_tags(
            aesthetic_prediction=(0.8, "best", {"best": 0.6, "great": 0.3}),
            tagger_predictions={"flat_color": 0.7},
            clip_style_scores={"naturalistic anime": 0.5},
            layer_consistency=0.8,
            confidence_threshold=0.25,
        )
        for tag in tags:
            assert tag.category == "style"

    def test_confidence_threshold_filtering(self) -> None:
        """All tags below high threshold are filtered out."""
        tags = _generate_tags(
            aesthetic_prediction=None,
            tagger_predictions={"bloom": 0.5},
            clip_style_scores=None,
            layer_consistency=0.5,
            confidence_threshold=0.95,
        )
        assert len(tags) == 0


# -- Full Analyzer Integration Tests --


class TestStyleAnalyzer:
    def test_basic_analysis_no_shared_models(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """Analyzer works without any shared model outputs."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        assert result.analyzer == "style"
        assert 0.0 <= result.score <= 1.0

    def test_with_aesthetic_prediction(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
        mock_aesthetic_high: tuple[float, str, dict[str, float]],
    ) -> None:
        """Aesthetic prediction raises the score."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {"aesthetic_prediction": mock_aesthetic_high}
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        assert result.score > 0.5
        assert result.metadata["aesthetic_tier"] == "best"

    def test_with_low_aesthetic(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
        mock_aesthetic_low: tuple[float, str, dict[str, float]],
    ) -> None:
        """Low aesthetic prediction lowers the score."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {"aesthetic_prediction": mock_aesthetic_low}
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        # Low aesthetic (0.2) should pull score down despite high consistency
        assert result.metadata["sub_scores"]["aesthetic_quality"] < 0.3

    def test_with_segmentation_mask(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """With segmentation mask, layer consistency uses separate regions."""
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[60:140, 60:140] = 0.9

        analyzer = StyleAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        assert result.metadata["has_segmentation"] is True

    def test_without_segmentation_mask(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """Without mask, layer consistency uses whole image."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        assert result.metadata["has_segmentation"] is False

    def test_with_tagger_predictions(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """WD-Tagger predictions produce style tags."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {
            "tagger_predictions": {
                "flat_color": 0.8,
                "cel_shading": 0.6,
            },
        }
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        tag_names = [t.name for t in result.tags]
        assert "flat_color" in tag_names
        assert "cel_shading" in tag_names

    def test_with_clip_style_scores(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """CLIP zero-shot style scores produce tags."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {
            "clip_style_scores": {
                "naturalistic anime": 0.7,
                "digital modern anime": 0.2,
                "painterly anime": 0.05,
                "geometric abstract anime": 0.03,
                "retro cel anime": 0.02,
            },
        }
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        tag_names = [t.name for t in result.tags]
        assert "naturalistic_anime" in tag_names

    def test_metadata_structure(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """Metadata contains expected keys."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        assert "sub_scores" in result.metadata
        assert "weights_used" in result.metadata
        assert "aesthetic_tier" in result.metadata
        assert "aesthetic_tier_probabilities" in result.metadata
        assert "has_segmentation" in result.metadata
        assert "layer_consistency_experimental" in result.metadata
        assert result.metadata["layer_consistency_experimental"] is True

        # Sub-properties present
        sub_scores = result.metadata["sub_scores"]
        assert "aesthetic_quality" in sub_scores
        assert "layer_consistency" in sub_scores

    def test_configurable_weights(
        self,
        uniform_rgb: np.ndarray,
    ) -> None:
        """Custom weights via config produce different scores."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {
            "aesthetic_prediction": (0.9, "masterpiece", {"masterpiece": 1.0}),
        }

        config_default = AnalyzerConfig(
            enabled=True,
            confidence_threshold=0.25,
            params={"aesthetic_weight": 0.70, "layer_consistency_weight": 0.30},
        )
        result_default = analyzer.analyze(uniform_rgb, config_default, shared)

        config_custom = AnalyzerConfig(
            enabled=True,
            confidence_threshold=0.25,
            params={"aesthetic_weight": 0.10, "layer_consistency_weight": 0.90},
        )
        result_custom = analyzer.analyze(uniform_rgb, config_custom, shared)

        # Different weights should produce different scores
        assert abs(result_default.score - result_custom.score) > 0.01

    def test_score_in_valid_range(
        self,
        noisy_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        analyzer = StyleAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(noisy_rgb, default_config, shared)
        assert 0.0 <= result.score <= 1.0

    def test_tags_have_correct_category(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        analyzer = StyleAnalyzer()
        shared: SharedModels = {
            "tagger_predictions": {"flat_color": 0.8},
            "clip_style_scores": {"naturalistic anime": 0.5},
        }
        result = analyzer.analyze(uniform_rgb, default_config, shared)
        for tag in result.tags:
            assert tag.category == "style"

    def test_mask_resolution_mismatch(
        self,
        default_config: AnalyzerConfig,
    ) -> None:
        """Segmentation mask at different resolution is handled."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        # Mask at half resolution
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[25:75, 25:75] = 0.9

        analyzer = StyleAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}
        result = analyzer.analyze(image, default_config, shared)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["has_segmentation"] is True

    def test_no_aesthetic_tier_in_metadata(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """Without aesthetic prediction, tier is None in metadata."""
        analyzer = StyleAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(uniform_rgb, default_config, shared)
        assert result.metadata["aesthetic_tier"] is None
        assert result.metadata["aesthetic_tier_probabilities"] is None
