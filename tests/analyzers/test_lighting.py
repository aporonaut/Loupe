"""Tests for the lighting analyzer."""

import numpy as np
import pytest

from loupe.analyzers.base import AnalyzerConfig, SharedModels
from loupe.analyzers.lighting import (
    LightingAnalyzer,
    _classify_directionality,
    _classify_tonality,
    _combine_scores,
    _generate_tags,
    measure_atmospheric_lighting,
    measure_contrast_ratio,
    measure_highlight_shadow_balance,
    measure_light_directionality,
    measure_rim_edge_lighting,
    measure_shadow_quality,
)

# -- Fixtures --


@pytest.fixture
def uniform_v() -> np.ndarray:
    """Uniform brightness — flat lighting."""
    return np.full((200, 200), 128, dtype=np.uint8)


@pytest.fixture
def high_contrast_v() -> np.ndarray:
    """Strong tonal range — left half dark, right half bright."""
    arr = np.zeros((200, 200), dtype=np.uint8)
    arr[:, :100] = 20
    arr[:, 100:] = 235
    return arr


@pytest.fixture
def low_contrast_v() -> np.ndarray:
    """Narrow tonal range — slight variation around midtone."""
    rng = np.random.default_rng(42)
    return (120 + rng.integers(0, 20, (200, 200))).astype(np.uint8)


@pytest.fixture
def top_lit_v() -> np.ndarray:
    """Top-lit image — bright top, dark bottom (gradient)."""
    col = np.linspace(240, 30, 200, dtype=np.uint8).reshape(-1, 1)
    return np.tile(col, (1, 200))


@pytest.fixture
def dark_image_v() -> np.ndarray:
    """Low-key image — mostly dark with some midtones."""
    arr = np.full((200, 200), 30, dtype=np.uint8)
    arr[80:120, 80:120] = 120  # Small midtone region
    return arr


@pytest.fixture
def bright_image_v() -> np.ndarray:
    """High-key image — mostly bright."""
    arr = np.full((200, 200), 220, dtype=np.uint8)
    arr[80:120, 80:120] = 180  # Slightly dimmer region
    return arr


@pytest.fixture
def center_character_mask() -> np.ndarray:
    """Character mask with center 40% foreground."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:140, 60:140] = 1
    return mask


@pytest.fixture
def rim_lit_image_v() -> np.ndarray:
    """Image with bright boundary ring around a dark center character."""
    # Dark background
    arr = np.full((200, 200), 40, dtype=np.uint8)
    # Character fill (mid-gray)
    arr[60:140, 60:140] = 100
    # Bright boundary ring around character (rim light)
    arr[58:62, 58:142] = 220
    arr[138:142, 58:142] = 220
    arr[58:142, 58:62] = 220
    arr[58:142, 138:142] = 220
    return arr


@pytest.fixture
def shadow_boundary_v() -> np.ndarray:
    """Image with clear shadow boundary — hard transition."""
    arr = np.full((200, 200), 180, dtype=np.uint8)
    # Sharp shadow region in bottom half
    arr[100:, :] = 40
    return arr


@pytest.fixture
def soft_shadow_v() -> np.ndarray:
    """Image with soft shadow transition — gradual gradient."""
    arr = np.zeros((200, 200), dtype=np.uint8)
    for y in range(200):
        val = int(180 - (140 * y / 200))
        arr[y, :] = max(0, min(255, val))
    return arr


@pytest.fixture
def bloom_image_v() -> np.ndarray:
    """Image with localized bright bloom regions."""
    arr = np.full((200, 200), 80, dtype=np.uint8)
    # Bright bloom spot in center
    y, x = np.ogrid[:200, :200]
    center_dist = np.sqrt((y - 100) ** 2 + (x - 100) ** 2)
    bloom = (255 * np.exp(-center_dist / 15.0)).astype(np.uint8)
    arr = np.maximum(arr, bloom)
    return arr


@pytest.fixture
def default_config() -> AnalyzerConfig:
    """Default lighting analyzer config."""
    return AnalyzerConfig(
        enabled=True,
        confidence_threshold=0.25,
    )


@pytest.fixture
def uniform_rgb() -> np.ndarray:
    """Uniform mid-gray RGB image."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def high_contrast_rgb() -> np.ndarray:
    """High-contrast RGB image — left dark, right bright."""
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :100] = 20
    arr[:, 100:] = 235
    return arr


@pytest.fixture
def rim_lit_rgb() -> np.ndarray:
    """RGB image with rim-lit character in center."""
    # Dark background
    arr = np.full((200, 200, 3), 40, dtype=np.uint8)
    # Character fill
    arr[60:140, 60:140] = 100
    # Bright boundary ring (rim light)
    arr[58:62, 58:142] = 220
    arr[138:142, 58:142] = 220
    arr[58:142, 58:62] = 220
    arr[58:142, 138:142] = 220
    return arr


# -- Contrast Ratio Tests --


class TestContrastRatio:
    def test_uniform_low_contrast(self, uniform_v: np.ndarray) -> None:
        score = measure_contrast_ratio(uniform_v)
        assert score < 0.05

    def test_high_contrast_image(self, high_contrast_v: np.ndarray) -> None:
        score = measure_contrast_ratio(high_contrast_v)
        assert score > 0.7

    def test_low_contrast_image(self, low_contrast_v: np.ndarray) -> None:
        score = measure_contrast_ratio(low_contrast_v)
        assert score < 0.15

    def test_uniform_lower_than_high_contrast(
        self, uniform_v: np.ndarray, high_contrast_v: np.ndarray
    ) -> None:
        assert measure_contrast_ratio(uniform_v) < measure_contrast_ratio(
            high_contrast_v
        )

    def test_score_in_range(self, top_lit_v: np.ndarray) -> None:
        score = measure_contrast_ratio(top_lit_v)
        assert 0.0 <= score <= 1.0


# -- Light Directionality Tests --


class TestLightDirectionality:
    def test_uniform_low_directionality(self, uniform_v: np.ndarray) -> None:
        score = measure_light_directionality(uniform_v)
        assert score < 0.1

    def test_top_lit_has_directionality(self, top_lit_v: np.ndarray) -> None:
        score = measure_light_directionality(top_lit_v)
        assert score > 0.3

    def test_uniform_lower_than_directional(
        self, uniform_v: np.ndarray, top_lit_v: np.ndarray
    ) -> None:
        assert measure_light_directionality(uniform_v) < measure_light_directionality(
            top_lit_v
        )

    def test_score_in_range(self, high_contrast_v: np.ndarray) -> None:
        score = measure_light_directionality(high_contrast_v)
        assert 0.0 <= score <= 1.0

    def test_tiny_image_returns_zero(self) -> None:
        tiny = np.full((2, 2), 128, dtype=np.uint8)
        assert measure_light_directionality(tiny) == 0.0


# -- Rim/Edge Lighting Tests --


class TestRimEdgeLighting:
    def test_no_mask_returns_zero(self, uniform_v: np.ndarray) -> None:
        score = measure_rim_edge_lighting(uniform_v, None)
        assert score == 0.0

    def test_empty_mask_returns_zero(self, uniform_v: np.ndarray) -> None:
        empty_mask = np.zeros((200, 200), dtype=np.uint8)
        score = measure_rim_edge_lighting(uniform_v, empty_mask)
        assert score == 0.0

    def test_rim_lit_image_scores_high(
        self,
        rim_lit_image_v: np.ndarray,
        center_character_mask: np.ndarray,
    ) -> None:
        score = measure_rim_edge_lighting(rim_lit_image_v, center_character_mask)
        assert score > 0.3

    def test_uniform_image_no_rim(
        self,
        uniform_v: np.ndarray,
        center_character_mask: np.ndarray,
    ) -> None:
        score = measure_rim_edge_lighting(uniform_v, center_character_mask)
        # Uniform brightness: no differential between boundary and background
        assert score < 0.15

    def test_score_in_range(
        self,
        rim_lit_image_v: np.ndarray,
        center_character_mask: np.ndarray,
    ) -> None:
        score = measure_rim_edge_lighting(rim_lit_image_v, center_character_mask)
        assert 0.0 <= score <= 1.0


# -- Shadow Quality Tests --


class TestShadowQuality:
    def test_hard_shadow_boundary(self, shadow_boundary_v: np.ndarray) -> None:
        score = measure_shadow_quality(shadow_boundary_v)
        # Hard shadow transition should score reasonably well
        assert score > 0.3

    def test_soft_shadow_gradient(self, soft_shadow_v: np.ndarray) -> None:
        score = measure_shadow_quality(soft_shadow_v)
        assert score > 0.2

    def test_uniform_low_shadow_quality(self, uniform_v: np.ndarray) -> None:
        score = measure_shadow_quality(uniform_v)
        # Uniform brightness: virtually no shadows
        assert score < 0.3

    def test_score_in_range(self, high_contrast_v: np.ndarray) -> None:
        score = measure_shadow_quality(high_contrast_v)
        assert 0.0 <= score <= 1.0


# -- Atmospheric Lighting Tests --


class TestAtmosphericLighting:
    def test_bloom_image_detected(self, bloom_image_v: np.ndarray) -> None:
        score = measure_atmospheric_lighting(bloom_image_v)
        assert score > 0.2

    def test_uniform_no_bloom(self, uniform_v: np.ndarray) -> None:
        score = measure_atmospheric_lighting(uniform_v)
        assert score < 0.05

    def test_uniform_lower_than_bloom(
        self, uniform_v: np.ndarray, bloom_image_v: np.ndarray
    ) -> None:
        assert measure_atmospheric_lighting(uniform_v) < measure_atmospheric_lighting(
            bloom_image_v
        )

    def test_score_in_range(self, bloom_image_v: np.ndarray) -> None:
        score = measure_atmospheric_lighting(bloom_image_v)
        assert 0.0 <= score <= 1.0


# -- Highlight/Shadow Balance Tests --


class TestHighlightShadowBalance:
    def test_dark_image_has_tonal_character(self, dark_image_v: np.ndarray) -> None:
        score = measure_highlight_shadow_balance(dark_image_v)
        # Low-key image: intentional darkness should score decently
        assert score > 0.3

    def test_bright_image_has_tonal_character(self, bright_image_v: np.ndarray) -> None:
        score = measure_highlight_shadow_balance(bright_image_v)
        assert score > 0.3

    def test_high_contrast_balanced(self, high_contrast_v: np.ndarray) -> None:
        score = measure_highlight_shadow_balance(high_contrast_v)
        # Has both shadows and highlights — should have some tonal character
        assert score > 0.3

    def test_score_in_range(self, uniform_v: np.ndarray) -> None:
        score = measure_highlight_shadow_balance(uniform_v)
        assert 0.0 <= score <= 1.0


# -- Directionality Classification Tests --


class TestDirectionalityClassification:
    def test_uniform_no_direction(self, uniform_v: np.ndarray) -> None:
        direction = _classify_directionality(uniform_v)
        assert direction is None

    def test_top_lit_direction(self, top_lit_v: np.ndarray) -> None:
        direction = _classify_directionality(top_lit_v)
        assert direction == "top"

    def test_left_lit_direction(self) -> None:
        """Left-lit image: bright left, dark right."""
        col = np.linspace(240, 30, 200, dtype=np.uint8).reshape(1, -1)
        left_lit = np.tile(col, (200, 1))
        direction = _classify_directionality(left_lit)
        assert direction == "left"

    def test_tiny_image_no_direction(self) -> None:
        tiny = np.full((2, 2), 128, dtype=np.uint8)
        assert _classify_directionality(tiny) is None


# -- Tonality Classification Tests --


class TestTonalityClassification:
    def test_dark_image_low_key(self, dark_image_v: np.ndarray) -> None:
        assert _classify_tonality(dark_image_v) == "low_key"

    def test_bright_image_high_key(self, bright_image_v: np.ndarray) -> None:
        assert _classify_tonality(bright_image_v) == "high_key"

    def test_uniform_balanced(self, uniform_v: np.ndarray) -> None:
        assert _classify_tonality(uniform_v) == "balanced"


# -- Score Combination Tests --


class TestCombineScores:
    def test_equal_weights(self) -> None:
        scores = {"a": 0.8, "b": 0.4}
        weights = {"a": 1.0, "b": 1.0}
        result = _combine_scores(scores, weights)
        assert result == pytest.approx(0.6, abs=0.01)

    def test_weighted_combination(self) -> None:
        scores = {"a": 1.0, "b": 0.0}
        weights = {"a": 0.75, "b": 0.25}
        result = _combine_scores(scores, weights)
        assert result == pytest.approx(0.75, abs=0.01)

    def test_zero_weights(self) -> None:
        scores = {"a": 0.5}
        weights = {"a": 0.0}
        assert _combine_scores(scores, weights) == 0.0

    def test_empty_scores(self) -> None:
        assert _combine_scores({}, {"a": 1.0}) == 0.0


# -- Tag Generation Tests --


class TestGenerateTags:
    def test_dramatic_lighting_tag(self) -> None:
        tags = _generate_tags(
            overall_score=0.8,
            sub_scores={"contrast_ratio": 0.5, "rim_edge_lighting": 0.0},
            light_direction=None,
            tagger_predictions=None,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "dramatic_lighting" in names

    def test_flat_lighting_tag(self) -> None:
        tags = _generate_tags(
            overall_score=0.15,
            sub_scores={"contrast_ratio": 0.5, "rim_edge_lighting": 0.0},
            light_direction=None,
            tagger_predictions=None,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "flat_lighting" in names

    def test_high_contrast_tag(self) -> None:
        tags = _generate_tags(
            overall_score=0.5,
            sub_scores={"contrast_ratio": 0.8, "rim_edge_lighting": 0.0},
            light_direction=None,
            tagger_predictions=None,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "high_contrast" in names

    def test_low_contrast_tag(self) -> None:
        tags = _generate_tags(
            overall_score=0.3,
            sub_scores={"contrast_ratio": 0.2, "rim_edge_lighting": 0.0},
            light_direction=None,
            tagger_predictions=None,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "low_contrast" in names

    def test_rim_lit_tag(self) -> None:
        tags = _generate_tags(
            overall_score=0.5,
            sub_scores={
                "contrast_ratio": 0.5,
                "rim_edge_lighting": 0.6,
                "atmospheric_lighting": 0.0,
            },
            light_direction=None,
            tagger_predictions=None,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "rim_lit" in names

    def test_rim_lit_with_tagger_crossref(self) -> None:
        """Rim light tag is boosted when WD-Tagger also detects rim_lighting."""
        tags = _generate_tags(
            overall_score=0.5,
            sub_scores={
                "contrast_ratio": 0.5,
                "rim_edge_lighting": 0.4,
                "atmospheric_lighting": 0.0,
            },
            light_direction=None,
            tagger_predictions={"rim_lighting": 0.7},
            confidence_threshold=0.25,
        )
        rim_tags = [t for t in tags if t.name == "rim_lit"]
        assert len(rim_tags) == 1
        assert rim_tags[0].confidence > 0.4  # boosted by tagger

    def test_directional_light_tag(self) -> None:
        tags = _generate_tags(
            overall_score=0.5,
            sub_scores={
                "contrast_ratio": 0.5,
                "rim_edge_lighting": 0.0,
                "light_directionality": 0.6,
            },
            light_direction="top",
            tagger_predictions=None,
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "directional_light" in names

    def test_tagger_supplementary_tags(self) -> None:
        """WD-Tagger lighting tags are passed through as supplementary."""
        tags = _generate_tags(
            overall_score=0.5,
            sub_scores={"contrast_ratio": 0.5, "rim_edge_lighting": 0.0},
            light_direction=None,
            tagger_predictions={
                "backlighting": 0.8,
                "lens_flare": 0.6,
                "unrelated_tag": 0.9,
            },
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "backlighting" in names
        assert "lens_flare" in names
        assert "unrelated_tag" not in names

    def test_tagger_below_threshold_filtered(self) -> None:
        """WD-Tagger tags below confidence threshold are excluded."""
        tags = _generate_tags(
            overall_score=0.5,
            sub_scores={"contrast_ratio": 0.5, "rim_edge_lighting": 0.0},
            light_direction=None,
            tagger_predictions={"sunlight": 0.1},
            confidence_threshold=0.25,
        )
        names = [t.name for t in tags]
        assert "sunlight" not in names

    def test_all_tags_have_lighting_category(self) -> None:
        tags = _generate_tags(
            overall_score=0.8,
            sub_scores={
                "contrast_ratio": 0.8,
                "rim_edge_lighting": 0.6,
                "shadow_quality": 0.7,
                "atmospheric_lighting": 0.5,
                "highlight_shadow_balance": 0.9,
                "light_directionality": 0.6,
            },
            light_direction="top",
            tagger_predictions={"backlighting": 0.7},
            confidence_threshold=0.25,
        )
        for tag in tags:
            assert tag.category == "lighting"

    def test_confidence_threshold_filtering(self) -> None:
        """Tags below the confidence threshold are filtered out."""
        tags = _generate_tags(
            overall_score=0.15,
            sub_scores={"contrast_ratio": 0.2, "rim_edge_lighting": 0.0},
            light_direction=None,
            tagger_predictions=None,
            confidence_threshold=0.9,
        )
        # flat_lighting has confidence 0.85, which is below 0.9 threshold
        assert len(tags) == 0


# -- Full Analyzer Integration Tests --


class TestLightingAnalyzer:
    def test_uniform_image_low_score(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(uniform_rgb, default_config, shared)
        assert result.score < 0.5
        assert result.analyzer == "lighting"

    def test_high_contrast_image(
        self,
        high_contrast_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(high_contrast_rgb, default_config, shared)
        assert result.score > 0.3

    def test_with_segmentation_mask(
        self,
        rim_lit_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """With segmentation mask, rim-light detection is active."""
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[60:140, 60:140] = 0.9

        analyzer = LightingAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}
        result = analyzer.analyze(rim_lit_rgb, default_config, shared)

        assert result.metadata["has_segmentation"] is True
        assert result.metadata["sub_scores"]["rim_edge_lighting"] > 0.0

    def test_without_segmentation_mask(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """Without segmentation mask, rim_edge_lighting is 0."""
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        assert result.metadata["has_segmentation"] is False
        assert result.metadata["sub_scores"]["rim_edge_lighting"] == 0.0

    def test_with_tagger_predictions(
        self,
        uniform_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """WD-Tagger predictions produce supplementary tags."""
        analyzer = LightingAnalyzer()
        shared: SharedModels = {
            "tagger_predictions": {
                "backlighting": 0.8,
                "lens_flare": 0.6,
            },
        }
        result = analyzer.analyze(uniform_rgb, default_config, shared)

        tag_names = [t.name for t in result.tags]
        assert "backlighting" in tag_names
        assert "lens_flare" in tag_names

    def test_metadata_structure(
        self,
        high_contrast_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """Metadata contains expected keys."""
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(high_contrast_rgb, default_config, shared)

        assert "sub_scores" in result.metadata
        assert "weights_used" in result.metadata
        assert "tonality" in result.metadata
        assert "light_direction" in result.metadata
        assert "has_segmentation" in result.metadata

        # All sub-properties present
        sub_scores = result.metadata["sub_scores"]
        expected_keys = {
            "contrast_ratio",
            "light_directionality",
            "rim_edge_lighting",
            "shadow_quality",
            "atmospheric_lighting",
            "highlight_shadow_balance",
        }
        assert set(sub_scores.keys()) == expected_keys

    def test_configurable_sub_weights(
        self,
        high_contrast_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        """Custom sub-property weights produce different scores."""
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}

        result_default = analyzer.analyze(high_contrast_rgb, default_config, shared)

        custom_config = AnalyzerConfig(
            enabled=True,
            confidence_threshold=0.25,
            params={"sub_weights": {"contrast_ratio": 1.0}},
        )
        result_custom = analyzer.analyze(high_contrast_rgb, custom_config, shared)

        assert result_default.score != result_custom.score

    def test_score_in_valid_range(
        self,
        high_contrast_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(high_contrast_rgb, default_config, shared)
        assert 0.0 <= result.score <= 1.0

    def test_tags_have_correct_category(
        self,
        high_contrast_rgb: np.ndarray,
        default_config: AnalyzerConfig,
    ) -> None:
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(high_contrast_rgb, default_config, shared)
        for tag in result.tags:
            assert tag.category == "lighting"

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

        analyzer = LightingAnalyzer()
        shared: SharedModels = {"segmentation_mask": mask}
        result = analyzer.analyze(image, default_config, shared)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["has_segmentation"] is True

    def test_tonality_in_metadata(
        self,
        default_config: AnalyzerConfig,
    ) -> None:
        """Tonality classification appears in metadata."""
        # Dark image → low_key
        dark_rgb = np.full((200, 200, 3), 30, dtype=np.uint8)
        analyzer = LightingAnalyzer()
        shared: SharedModels = {}
        result = analyzer.analyze(dark_rgb, default_config, shared)
        assert result.metadata["tonality"] == "low_key"
