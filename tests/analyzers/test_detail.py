"""Tests for the detail analyzer."""

import numpy as np
import pytest

from loupe.analyzers.base import AnalyzerConfig, SharedModels
from loupe.analyzers.detail import (
    DetailAnalyzer,
    _combine_sub_scores,
    _generate_tags,
    _get_region_masks,
    _measure_region,
    measure_edge_density,
    measure_line_work_quality,
    measure_rendering_clarity,
    measure_shading_granularity,
    measure_spatial_frequency,
    measure_texture_richness,
)

# -- Fixtures --


@pytest.fixture
def flat_gray() -> np.ndarray:
    """Solid gray image — minimal detail."""
    return np.full((200, 200), 128, dtype=np.uint8)


@pytest.fixture
def noisy_gray() -> np.ndarray:
    """Random noise image — high-frequency content everywhere."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 200), dtype=np.uint8)


@pytest.fixture
def gradient_gray() -> np.ndarray:
    """Horizontal gradient — moderate detail."""
    row = np.linspace(0, 255, 200, dtype=np.uint8)
    return np.tile(row, (200, 1))


@pytest.fixture
def edge_image_gray() -> np.ndarray:
    """Image with sharp vertical edges — good line work."""
    arr = np.full((200, 200), 128, dtype=np.uint8)
    # Create multiple sharp vertical edges
    for x in range(20, 200, 40):
        arr[:, x : x + 2] = 30  # Dark lines on mid-gray
    return arr


@pytest.fixture
def multi_tone_gray() -> np.ndarray:
    """Image with multiple distinct tonal bands — complex shading."""
    arr = np.zeros((200, 200), dtype=np.uint8)
    for i, val in enumerate([30, 80, 120, 160, 200, 240]):
        y0 = i * 33
        y1 = min(y0 + 33, 200)
        arr[y0:y1, :] = val
    return arr


@pytest.fixture
def flat_rgb() -> np.ndarray:
    """Solid color RGB image — minimal detail."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def detailed_rgb() -> np.ndarray:
    """Random-noise RGB image — high detail."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)


@pytest.fixture
def center_mask() -> np.ndarray:
    """Bool mask with center 40% foreground."""
    mask = np.zeros((200, 200), dtype=bool)
    mask[60:140, 60:140] = True
    return mask


@pytest.fixture
def full_mask() -> np.ndarray:
    """Bool mask that is entirely foreground (>90%)."""
    mask = np.ones((200, 200), dtype=bool)
    return mask


@pytest.fixture
def default_config() -> AnalyzerConfig:
    """Default detail analyzer config."""
    return AnalyzerConfig(
        enabled=True,
        confidence_threshold=0.25,
        params={"bg_weight": 0.6, "char_weight": 0.4},
    )


# -- Edge Density Tests --


class TestEdgeDensity:
    def test_flat_image_low_score(self, flat_gray: np.ndarray) -> None:
        score = measure_edge_density(flat_gray)
        assert score < 0.05

    def test_noisy_image_high_score(self, noisy_gray: np.ndarray) -> None:
        score = measure_edge_density(noisy_gray)
        assert score > 0.3

    def test_flat_lower_than_noisy(
        self, flat_gray: np.ndarray, noisy_gray: np.ndarray
    ) -> None:
        flat_score = measure_edge_density(flat_gray)
        noisy_score = measure_edge_density(noisy_gray)
        assert flat_score < noisy_score

    def test_with_mask(self, noisy_gray: np.ndarray, center_mask: np.ndarray) -> None:
        score = measure_edge_density(noisy_gray, center_mask)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self, gradient_gray: np.ndarray) -> None:
        score = measure_edge_density(gradient_gray)
        assert 0.0 <= score <= 1.0


# -- Spatial Frequency Tests --


class TestSpatialFrequency:
    def test_flat_image_low_score(self, flat_gray: np.ndarray) -> None:
        score = measure_spatial_frequency(flat_gray)
        assert score < 0.1

    def test_noisy_image_high_score(self, noisy_gray: np.ndarray) -> None:
        score = measure_spatial_frequency(noisy_gray)
        assert score > 0.3

    def test_flat_lower_than_noisy(
        self, flat_gray: np.ndarray, noisy_gray: np.ndarray
    ) -> None:
        assert measure_spatial_frequency(flat_gray) < measure_spatial_frequency(
            noisy_gray
        )

    def test_with_mask(self, noisy_gray: np.ndarray, center_mask: np.ndarray) -> None:
        score = measure_spatial_frequency(noisy_gray, center_mask)
        assert 0.0 <= score <= 1.0

    def test_tiny_region_falls_back_to_full_image(self, noisy_gray: np.ndarray) -> None:
        # Mask with too few pixels (<100) falls back to full-image analysis
        mask = np.zeros((200, 200), dtype=bool)
        mask[0, 0] = True
        score_masked = measure_spatial_frequency(noisy_gray, mask)
        score_full = measure_spatial_frequency(noisy_gray, None)
        assert score_masked == pytest.approx(score_full)


# -- Texture Richness Tests --


class TestTextureRichness:
    def test_flat_image_low_score(self, flat_gray: np.ndarray) -> None:
        score = measure_texture_richness(flat_gray)
        assert score < 0.3

    def test_noisy_image_higher_score(self, noisy_gray: np.ndarray) -> None:
        score = measure_texture_richness(noisy_gray)
        assert score > 0.2

    def test_flat_lower_than_noisy(
        self, flat_gray: np.ndarray, noisy_gray: np.ndarray
    ) -> None:
        assert measure_texture_richness(flat_gray) < measure_texture_richness(
            noisy_gray
        )

    def test_with_mask(self, noisy_gray: np.ndarray, center_mask: np.ndarray) -> None:
        score = measure_texture_richness(noisy_gray, center_mask)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self, gradient_gray: np.ndarray) -> None:
        score = measure_texture_richness(gradient_gray)
        assert 0.0 <= score <= 1.0


# -- Shading Granularity Tests --


class TestShadingGranularity:
    def test_flat_image_low_score(self, flat_gray: np.ndarray) -> None:
        score = measure_shading_granularity(flat_gray)
        # Single tone — low peak count
        assert score < 0.3

    def test_multi_tone_higher_score(self, multi_tone_gray: np.ndarray) -> None:
        score = measure_shading_granularity(multi_tone_gray)
        assert score > 0.3

    def test_flat_lower_than_multi_tone(
        self, flat_gray: np.ndarray, multi_tone_gray: np.ndarray
    ) -> None:
        assert measure_shading_granularity(flat_gray) < measure_shading_granularity(
            multi_tone_gray
        )

    def test_with_mask(
        self, multi_tone_gray: np.ndarray, center_mask: np.ndarray
    ) -> None:
        score = measure_shading_granularity(multi_tone_gray, center_mask)
        assert 0.0 <= score <= 1.0

    def test_too_few_pixels_returns_zero(self) -> None:
        tiny = np.full((5, 5), 128, dtype=np.uint8)
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        score = measure_shading_granularity(tiny, mask)
        assert score == 0.0


# -- Line Work Quality Tests --


class TestLineWorkQuality:
    def test_flat_image_low_score(self, flat_gray: np.ndarray) -> None:
        score = measure_line_work_quality(flat_gray)
        assert score < 0.1

    def test_edge_image_higher_score(self, edge_image_gray: np.ndarray) -> None:
        score = measure_line_work_quality(edge_image_gray)
        assert score > 0.2

    def test_flat_lower_than_edges(
        self, flat_gray: np.ndarray, edge_image_gray: np.ndarray
    ) -> None:
        assert measure_line_work_quality(flat_gray) < measure_line_work_quality(
            edge_image_gray
        )

    def test_with_mask(
        self, edge_image_gray: np.ndarray, center_mask: np.ndarray
    ) -> None:
        score = measure_line_work_quality(edge_image_gray, center_mask)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self, noisy_gray: np.ndarray) -> None:
        score = measure_line_work_quality(noisy_gray)
        assert 0.0 <= score <= 1.0


# -- Rendering Clarity Tests --


class TestRenderingClarity:
    def test_flat_image_low_score(self, flat_gray: np.ndarray) -> None:
        score = measure_rendering_clarity(flat_gray)
        assert score < 0.1

    def test_noisy_image_higher_score(self, noisy_gray: np.ndarray) -> None:
        score = measure_rendering_clarity(noisy_gray)
        assert score > 0.3

    def test_flat_lower_than_noisy(
        self, flat_gray: np.ndarray, noisy_gray: np.ndarray
    ) -> None:
        assert measure_rendering_clarity(flat_gray) < measure_rendering_clarity(
            noisy_gray
        )

    def test_with_mask(self, noisy_gray: np.ndarray, center_mask: np.ndarray) -> None:
        score = measure_rendering_clarity(noisy_gray, center_mask)
        assert 0.0 <= score <= 1.0


# -- Region Separation Tests --


class TestRegionSeparation:
    def test_no_segmentation_all_background(self) -> None:
        shared: SharedModels = {}
        fg, bg, ratio = _get_region_masks(shared, 100, 100)
        assert ratio == 0.0
        assert np.all(bg)
        assert not np.any(fg)

    def test_with_segmentation_mask(self) -> None:
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 0.9  # Above threshold
        shared: SharedModels = {"segmentation_mask": mask}
        fg, bg, ratio = _get_region_masks(shared, 100, 100)
        assert 0.3 < ratio < 0.4  # ~36% foreground
        assert np.sum(fg) > 0
        assert np.sum(bg) > 0

    def test_mask_resizing(self) -> None:
        # Mask at different resolution than target
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 0.9
        shared: SharedModels = {"segmentation_mask": mask}
        fg, bg, ratio = _get_region_masks(shared, 100, 100)
        assert fg.shape == (100, 100)
        assert bg.shape == (100, 100)
        assert ratio > 0.0

    def test_full_foreground(self) -> None:
        mask = np.ones((100, 100), dtype=np.float32) * 0.9
        shared: SharedModels = {"segmentation_mask": mask}
        _fg, _bg, ratio = _get_region_masks(shared, 100, 100)
        assert ratio > 0.9

    def test_measure_region_returns_all_keys(self, noisy_gray: np.ndarray) -> None:
        scores = _measure_region(noisy_gray, None)
        expected_keys = {
            "edge_density",
            "spatial_frequency",
            "texture_richness",
            "shading_granularity",
            "line_work_quality",
            "rendering_clarity",
        }
        assert set(scores.keys()) == expected_keys
        for val in scores.values():
            assert 0.0 <= val <= 1.0


# -- Score Combination Tests --


class TestScoreCombination:
    def test_equal_weights(self) -> None:
        sub_scores = {"a": 0.5, "b": 0.5}
        weights = {"a": 1.0, "b": 1.0}
        assert _combine_sub_scores(sub_scores, weights) == pytest.approx(0.5)

    def test_unequal_weights(self) -> None:
        sub_scores = {"a": 1.0, "b": 0.0}
        weights = {"a": 0.75, "b": 0.25}
        assert _combine_sub_scores(sub_scores, weights) == pytest.approx(0.75)

    def test_zero_weights(self) -> None:
        sub_scores = {"a": 1.0}
        weights: dict[str, float] = {}
        assert _combine_sub_scores(sub_scores, weights) == 0.0

    def test_clamps_to_unit(self) -> None:
        sub_scores = {"a": 1.0}
        weights = {"a": 1.0}
        assert 0.0 <= _combine_sub_scores(sub_scores, weights) <= 1.0


# -- Tag Generation Tests --


class TestTagGeneration:
    def test_high_detail_tag(self) -> None:
        tags = _generate_tags(
            overall_score=0.8,
            bg_scores=None,
            char_scores=None,
            combined_scores={},
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "high_detail" in names

    def test_no_high_detail_tag_when_low(self) -> None:
        tags = _generate_tags(
            overall_score=0.3,
            bg_scores=None,
            char_scores=None,
            combined_scores={},
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "high_detail" not in names

    def test_rich_background_tag(self) -> None:
        bg_scores = {
            "edge_density": 0.7,
            "spatial_frequency": 0.7,
            "texture_richness": 0.7,
            "shading_granularity": 0.7,
            "line_work_quality": 0.7,
            "rendering_clarity": 0.7,
        }
        tags = _generate_tags(
            overall_score=0.7,
            bg_scores=bg_scores,
            char_scores=None,
            combined_scores=bg_scores,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "rich_background" in names

    def test_detailed_character_tag(self) -> None:
        char_scores = {
            "edge_density": 0.8,
            "spatial_frequency": 0.7,
            "texture_richness": 0.6,
            "shading_granularity": 0.7,
            "line_work_quality": 0.8,
            "rendering_clarity": 0.7,
        }
        tags = _generate_tags(
            overall_score=0.7,
            bg_scores=None,
            char_scores=char_scores,
            combined_scores=char_scores,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "detailed_character" in names

    def test_sub_property_tags(self) -> None:
        combined = {
            "rendering_clarity": 0.8,
            "shading_granularity": 0.7,
            "line_work_quality": 0.9,
        }
        tags = _generate_tags(
            overall_score=0.8,
            bg_scores=None,
            char_scores=None,
            combined_scores=combined,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "sharp_rendering" in names
        assert "complex_shading" in names
        assert "fine_line_work" in names

    def test_confidence_threshold_filters(self) -> None:
        tags = _generate_tags(
            overall_score=0.8,
            bg_scores=None,
            char_scores=None,
            combined_scores={"rendering_clarity": 0.2},
            confidence_threshold=0.5,
        )
        # 0.2 clarity < 0.6 threshold for tag generation, so no sharp_rendering
        # 0.8 overall >= 0.7 but must also pass confidence_threshold=0.5
        names = [t.name for t in tags]
        assert "sharp_rendering" not in names
        # high_detail: confidence=0.8 >= threshold=0.5
        assert "high_detail" in names

    def test_all_tags_have_detail_category(self) -> None:
        combined = {
            "rendering_clarity": 0.8,
            "shading_granularity": 0.7,
            "line_work_quality": 0.9,
        }
        tags = _generate_tags(
            overall_score=0.8,
            bg_scores=None,
            char_scores=None,
            combined_scores=combined,
            confidence_threshold=0.25,
        )
        for tag in tags:
            assert tag.category == "detail"


# -- Full Analyzer Integration Tests --


class TestDetailAnalyzer:
    def test_protocol_compliance(self) -> None:
        """Analyzer satisfies BaseAnalyzer protocol."""
        analyzer = DetailAnalyzer()
        assert analyzer.name == "detail"
        assert hasattr(analyzer, "analyze")

    def test_flat_image_low_score(
        self, flat_rgb: np.ndarray, default_config: AnalyzerConfig
    ) -> None:
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(flat_rgb, default_config, shared)
        assert result.analyzer == "detail"
        assert result.score < 0.2

    def test_detailed_image_higher_score(
        self, detailed_rgb: np.ndarray, default_config: AnalyzerConfig
    ) -> None:
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(detailed_rgb, default_config, shared)
        assert result.score > 0.3

    def test_flat_lower_than_detailed(
        self,
        flat_rgb: np.ndarray,
        detailed_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}
        flat_result = analyzer.analyze(flat_rgb, default_config, shared)
        detail_result = analyzer.analyze(detailed_rgb, default_config, shared)
        assert flat_result.score < detail_result.score

    def test_metadata_structure(
        self, detailed_rgb: np.ndarray, default_config: AnalyzerConfig
    ) -> None:
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(detailed_rgb, default_config, shared)

        assert "sub_scores" in result.metadata
        assert "weights_used" in result.metadata
        assert "region_weights" in result.metadata
        assert "character_ratio" in result.metadata
        assert "analysis_resolution" in result.metadata

        # All 6 sub-property keys present
        sub_scores = result.metadata["sub_scores"]
        expected_keys = {
            "edge_density",
            "spatial_frequency",
            "texture_richness",
            "shading_granularity",
            "line_work_quality",
            "rendering_clarity",
        }
        assert set(sub_scores.keys()) == expected_keys

    def test_no_segmentation_all_background(
        self, detailed_rgb: np.ndarray, default_config: AnalyzerConfig
    ) -> None:
        """Without segmentation mask, entire image is treated as background."""
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(detailed_rgb, default_config, shared)

        assert result.metadata["character_ratio"] == 0.0
        assert result.metadata["region_weights"]["background"] == 1.0
        assert result.metadata["region_weights"]["character"] == 0.0
        assert "bg_sub_scores" in result.metadata
        assert "char_sub_scores" not in result.metadata

    def test_with_segmentation_mask_both_regions(
        self, default_config: AnalyzerConfig
    ) -> None:
        """With segmentation mask, both regions are measured."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        mask = np.zeros((200, 200), dtype=np.float32)
        mask[50:150, 50:150] = 0.9  # Center is foreground

        analyzer = DetailAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}
        result = analyzer.analyze(image, default_config, shared)

        assert result.metadata["character_ratio"] > 0.0
        assert "bg_sub_scores" in result.metadata
        assert "char_sub_scores" in result.metadata
        assert result.metadata["region_weights"]["background"] == 0.6
        assert result.metadata["region_weights"]["character"] == 0.4

    def test_dominant_character_region(self, default_config: AnalyzerConfig) -> None:
        """Character >90% of frame: 100% character weight."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        # Mask covering >90% of image
        mask = np.ones((200, 200), dtype=np.float32) * 0.9

        analyzer = DetailAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}
        result = analyzer.analyze(image, default_config, shared)

        assert result.metadata["region_weights"]["background"] == 0.0
        assert result.metadata["region_weights"]["character"] == 1.0
        assert "char_sub_scores" in result.metadata

    def test_configurable_sub_weights(
        self, detailed_rgb: np.ndarray, default_config: AnalyzerConfig
    ) -> None:
        """Custom sub-property weights produce different scores."""
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}

        # Default weights
        result_default = analyzer.analyze(detailed_rgb, default_config, shared)

        # Custom weights: heavily favor edge density
        custom_config = AnalyzerConfig(
            enabled=True,
            confidence_threshold=0.25,
            params={
                "bg_weight": 0.6,
                "char_weight": 0.4,
                "sub_weights": {"edge_density": 1.0},
            },
        )
        result_custom = analyzer.analyze(detailed_rgb, custom_config, shared)

        # Scores should differ (different weighting)
        assert result_default.score != result_custom.score

    def test_configurable_region_weights(self, default_config: AnalyzerConfig) -> None:
        """Custom region weights are respected."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        mask = np.zeros((200, 200), dtype=np.float32)
        mask[50:150, 50:150] = 0.9

        analyzer = DetailAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}

        # Custom: 80% bg, 20% char
        custom_config = AnalyzerConfig(
            enabled=True,
            confidence_threshold=0.25,
            params={"bg_weight": 0.8, "char_weight": 0.2},
        )
        result = analyzer.analyze(image, custom_config, shared)
        assert result.metadata["region_weights"]["background"] == 0.8
        assert result.metadata["region_weights"]["character"] == 0.2

    def test_score_in_valid_range(
        self, detailed_rgb: np.ndarray, default_config: AnalyzerConfig
    ) -> None:
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(detailed_rgb, default_config, shared)
        assert 0.0 <= result.score <= 1.0

    def test_tags_have_correct_category(
        self, detailed_rgb: np.ndarray, default_config: AnalyzerConfig
    ) -> None:
        analyzer = DetailAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(detailed_rgb, default_config, shared)
        for tag in result.tags:
            assert tag.category == "detail"

    def test_mask_resolution_mismatch(self, default_config: AnalyzerConfig) -> None:
        """Segmentation mask at different resolution is handled."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        # Mask at half resolution
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[25:75, 25:75] = 0.9

        analyzer = DetailAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}
        result = analyzer.analyze(image, default_config, shared)
        assert result.score > 0.0
        assert result.metadata["character_ratio"] > 0.0
