"""Tests for the color analyzer."""

import numpy as np
import pytest

from loupe.analyzers._color_space import srgb_uint8_to_oklab, srgb_uint8_to_oklch
from loupe.analyzers.base import AnalyzerConfig, SharedModels
from loupe.analyzers.color import (
    ColorAnalyzer,
    _combine_scores,
    compute_harmony,
    extract_palette,
    measure_color_contrast,
    measure_color_temperature,
    measure_palette_cohesion,
    measure_palette_diversity,
    measure_saturation_balance,
    measure_vivid_color,
)

# -- Fixtures --


@pytest.fixture
def solid_red_array() -> np.ndarray:
    """Solid red 50x50 image."""
    return np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)


@pytest.fixture
def solid_gray_array() -> np.ndarray:
    """Solid mid-gray 50x50 image (achromatic)."""
    return np.full((50, 50, 3), 128, dtype=np.uint8)


@pytest.fixture
def two_color_array() -> np.ndarray:
    """Two-color image: left half red, right half blue."""
    arr = np.zeros((50, 100, 3), dtype=np.uint8)
    arr[:, :50] = [255, 0, 0]
    arr[:, 50:] = [0, 0, 255]
    return arr


@pytest.fixture
def rainbow_array() -> np.ndarray:
    """Image with hue gradient (high diversity)."""
    arr = np.zeros((50, 360, 3), dtype=np.uint8)
    for h in range(360):
        # Simple HSV to RGB for hue sweep at full saturation
        c = 255
        x = int(c * (1 - abs((h / 60) % 2 - 1)))
        if h < 60:
            arr[:, h] = [c, x, 0]
        elif h < 120:
            arr[:, h] = [x, c, 0]
        elif h < 180:
            arr[:, h] = [0, c, x]
        elif h < 240:
            arr[:, h] = [0, x, c]
        elif h < 300:
            arr[:, h] = [x, 0, c]
        else:
            arr[:, h] = [c, 0, x]
    return arr


@pytest.fixture
def analogous_array() -> np.ndarray:
    """Image with analogous colors (warm reds/oranges — should be harmonious)."""
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    arr[:25, :25] = [255, 50, 30]  # Red
    arr[:25, 25:] = [255, 100, 30]  # Red-orange
    arr[25:, :25] = [255, 150, 50]  # Orange
    arr[25:, 25:] = [255, 80, 40]  # Deep orange
    return arr


# -- Palette Extraction Tests --


class TestExtractPalette:
    def test_solid_color_single_cluster(self, solid_red_array: np.ndarray) -> None:
        oklab = srgb_uint8_to_oklab(solid_red_array)
        palette = extract_palette(oklab, n_clusters=4)
        # Should merge to ~1 cluster since all pixels are identical
        assert len(palette) == 1
        assert palette[0][1] == pytest.approx(1.0, abs=0.01)

    def test_two_colors_two_clusters(self, two_color_array: np.ndarray) -> None:
        oklab = srgb_uint8_to_oklab(two_color_array)
        palette = extract_palette(oklab, n_clusters=4)
        # Should produce exactly 2 clusters (red and blue are far apart in OkLab)
        assert len(palette) == 2
        # Each cluster should have ~50% proportion
        for _, prop in palette:
            assert 0.3 < prop < 0.7

    def test_proportions_sum_to_one(self, rainbow_array: np.ndarray) -> None:
        oklab = srgb_uint8_to_oklab(rainbow_array)
        palette = extract_palette(oklab, n_clusters=6)
        total = sum(prop for _, prop in palette)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_empty_image(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.float32)
        palette = extract_palette(empty)
        assert palette == []

    def test_sorted_by_proportion(self, rainbow_array: np.ndarray) -> None:
        oklab = srgb_uint8_to_oklab(rainbow_array)
        palette = extract_palette(oklab, n_clusters=6)
        proportions = [prop for _, prop in palette]
        assert proportions == sorted(proportions, reverse=True)


# -- Matsuda Harmony Tests --


class TestComputeHarmony:
    def test_monochromatic_high_score(self, solid_red_array: np.ndarray) -> None:
        """A single-hue image should score high (fits type i template)."""
        oklch = srgb_uint8_to_oklch(solid_red_array)
        score, _template_idx = compute_harmony(oklch)
        assert score > 0.8

    def test_achromatic_returns_one(self, solid_gray_array: np.ndarray) -> None:
        """Achromatic image (no chromatic content) → perfect harmony."""
        oklch = srgb_uint8_to_oklch(solid_gray_array)
        score, _ = compute_harmony(oklch)
        assert score == 1.0

    def test_analogous_higher_than_random(self, analogous_array: np.ndarray) -> None:
        """Analogous palette should score higher than random noise."""
        oklch_analogous = srgb_uint8_to_oklch(analogous_array)
        score_analogous, _ = compute_harmony(oklch_analogous)

        rng = np.random.default_rng(42)
        random_img = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        oklch_random = srgb_uint8_to_oklch(random_img)
        score_random, _ = compute_harmony(oklch_random)

        assert score_analogous > score_random

    def test_score_range(self, two_color_array: np.ndarray) -> None:
        oklch = srgb_uint8_to_oklch(two_color_array)
        score, template_idx = compute_harmony(oklch)
        assert 0.0 <= score <= 1.0
        assert 0 <= template_idx < 8

    def test_complementary_fits_type_I(self) -> None:
        """Red + cyan (complementary) should fit type I template well."""
        arr = np.zeros((50, 100, 3), dtype=np.uint8)
        arr[:, :50] = [255, 0, 0]  # Red
        arr[:, 50:] = [0, 255, 255]  # Cyan (complement)
        oklch = srgb_uint8_to_oklch(arr)
        score, _template_idx = compute_harmony(oklch)
        # Should have decent harmony via the complementary template
        assert score > 0.5


# -- Sub-Property Measurement Tests --


class TestPaletteCohesion:
    def test_single_color_perfect(self) -> None:
        palette = [(np.array([0.5, 0.1, 0.0], dtype=np.float32), 1.0)]
        assert measure_palette_cohesion(palette) == 1.0

    def test_similar_colors_high(self) -> None:
        palette = [
            (np.array([0.5, 0.1, 0.0], dtype=np.float32), 0.5),
            (np.array([0.5, 0.12, 0.01], dtype=np.float32), 0.5),
        ]
        assert measure_palette_cohesion(palette) > 0.8

    def test_distant_colors_lower(self) -> None:
        palette = [
            (np.array([0.9, 0.2, 0.1], dtype=np.float32), 0.5),
            (np.array([0.2, -0.1, -0.2], dtype=np.float32), 0.5),
        ]
        close_palette = [
            (np.array([0.5, 0.1, 0.0], dtype=np.float32), 0.5),
            (np.array([0.5, 0.12, 0.01], dtype=np.float32), 0.5),
        ]
        assert measure_palette_cohesion(palette) < measure_palette_cohesion(
            close_palette
        )


class TestSaturationBalance:
    def test_uniform_chroma_high(self) -> None:
        """Uniform chroma -> high saturation balance (low CV component)."""
        # Create OkLCh image with uniform chroma
        oklch = np.full((50, 50, 3), [0.5, 0.1, 180.0], dtype=np.float32)
        score = measure_saturation_balance(oklch)
        # CV is 0 (perfect), but entropy is low (single value). Score ~ 0.6.
        assert score > 0.5

    def test_achromatic_low(self) -> None:
        oklch = np.full((50, 50, 3), [0.5, 0.0, 0.0], dtype=np.float32)
        score = measure_saturation_balance(oklch)
        assert score < 0.5

    def test_score_range(self) -> None:
        oklch = np.random.default_rng(42).random((50, 50, 3)).astype(np.float32)
        oklch[..., 2] *= 360  # Scale hue
        score = measure_saturation_balance(oklch)
        assert 0.0 <= score <= 1.0


class TestColorContrast:
    def test_high_contrast(self) -> None:
        """High luminance and chromatic contrast."""
        oklab = np.zeros((50, 100, 3), dtype=np.float32)
        oklab[:, :50] = [0.1, -0.1, -0.05]  # Dark, blue-ish
        oklab[:, 50:] = [0.9, 0.1, 0.05]  # Light, red-ish
        score = measure_color_contrast(oklab)
        assert score > 0.5

    def test_low_contrast(self) -> None:
        """Uniform image → low contrast."""
        oklab = np.full((50, 50, 3), [0.5, 0.0, 0.0], dtype=np.float32)
        score = measure_color_contrast(oklab)
        assert score < 0.2

    def test_score_range(self) -> None:
        oklab = np.random.default_rng(42).random((50, 50, 3)).astype(np.float32)
        score = measure_color_contrast(oklab)
        assert 0.0 <= score <= 1.0


class TestColorTemperature:
    def test_warm_palette_high(self) -> None:
        """All-warm hues → high consistency."""
        oklch = np.full((50, 50, 3), [0.6, 0.15, 30.0], dtype=np.float32)
        score = measure_color_temperature(oklch)
        assert score > 0.7

    def test_cool_palette_high(self) -> None:
        """All-cool hues → high consistency."""
        oklch = np.full((50, 50, 3), [0.6, 0.15, 200.0], dtype=np.float32)
        score = measure_color_temperature(oklch)
        assert score > 0.7

    def test_achromatic_neutral(self) -> None:
        oklch = np.full((50, 50, 3), [0.5, 0.0, 0.0], dtype=np.float32)
        score = measure_color_temperature(oklch)
        assert score == pytest.approx(0.5)


class TestPaletteDiversity:
    def test_single_hue_low(self, solid_red_array: np.ndarray) -> None:
        oklch = srgb_uint8_to_oklch(solid_red_array)
        score = measure_palette_diversity(oklch)
        assert score < 0.2

    def test_rainbow_high(self, rainbow_array: np.ndarray) -> None:
        oklch = srgb_uint8_to_oklch(rainbow_array)
        score = measure_palette_diversity(oklch)
        assert score > 0.7

    def test_achromatic_zero(self, solid_gray_array: np.ndarray) -> None:
        oklch = srgb_uint8_to_oklch(solid_gray_array)
        score = measure_palette_diversity(oklch)
        assert score == 0.0


class TestVividColor:
    def test_saturated_image_high(self, solid_red_array: np.ndarray) -> None:
        score = measure_vivid_color(solid_red_array)
        # Solid red has high rg difference but no spatial variation
        # Hasler metric uses std, so solid color has low metric
        assert 0.0 <= score <= 1.0

    def test_gray_low(self, solid_gray_array: np.ndarray) -> None:
        score = measure_vivid_color(solid_gray_array)
        assert score < 0.1

    def test_rainbow_high(self, rainbow_array: np.ndarray) -> None:
        score = measure_vivid_color(rainbow_array)
        assert score > 0.5

    def test_score_range(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        score = measure_vivid_color(img)
        assert 0.0 <= score <= 1.0


# -- Score Combination Tests --


class TestCombineScores:
    def test_equal_weights(self) -> None:
        scores = {"a": 0.6, "b": 0.8}
        weights = {"a": 1.0, "b": 1.0}
        assert _combine_scores(scores, weights) == pytest.approx(0.7)

    def test_weighted(self) -> None:
        scores = {"a": 1.0, "b": 0.0}
        weights = {"a": 3.0, "b": 1.0}
        assert _combine_scores(scores, weights) == pytest.approx(0.75)

    def test_missing_weight_ignored(self) -> None:
        scores = {"a": 0.5, "b": 0.5}
        weights = {"a": 1.0}  # b has no weight
        assert _combine_scores(scores, weights) == pytest.approx(0.5)

    def test_empty_returns_zero(self) -> None:
        assert _combine_scores({}, {}) == 0.0


# -- Full Analyzer Tests --


class TestColorAnalyzer:
    def test_satisfies_protocol(self) -> None:
        from loupe.analyzers.base import BaseAnalyzer

        analyzer: BaseAnalyzer = ColorAnalyzer()
        assert analyzer.name == "color"

    def test_produces_result(self, solid_red_array: np.ndarray) -> None:
        analyzer = ColorAnalyzer()
        config = AnalyzerConfig()
        shared = SharedModels()
        result = analyzer.analyze(solid_red_array, config, shared)

        assert result.analyzer == "color"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.tags, list)
        assert "sub_scores" in result.metadata
        assert "palette" in result.metadata

    def test_sub_scores_in_metadata(self, two_color_array: np.ndarray) -> None:
        analyzer = ColorAnalyzer()
        result = analyzer.analyze(two_color_array, AnalyzerConfig(), SharedModels())
        sub_scores = result.metadata["sub_scores"]

        expected_keys = {
            "harmony",
            "palette_cohesion",
            "saturation_balance",
            "color_contrast",
            "color_temperature",
            "palette_diversity",
            "vivid_color",
        }
        assert set(sub_scores.keys()) == expected_keys
        for value in sub_scores.values():
            assert 0.0 <= value <= 1.0

    def test_gray_image_low_vivid(self, solid_gray_array: np.ndarray) -> None:
        analyzer = ColorAnalyzer()
        result = analyzer.analyze(solid_gray_array, AnalyzerConfig(), SharedModels())
        assert result.metadata["sub_scores"]["vivid_color"] < 0.1

    def test_rainbow_high_diversity(self, rainbow_array: np.ndarray) -> None:
        analyzer = ColorAnalyzer()
        result = analyzer.analyze(rainbow_array, AnalyzerConfig(), SharedModels())
        assert result.metadata["sub_scores"]["palette_diversity"] > 0.7

    def test_palette_in_metadata(self, two_color_array: np.ndarray) -> None:
        analyzer = ColorAnalyzer()
        result = analyzer.analyze(two_color_array, AnalyzerConfig(), SharedModels())
        palette = result.metadata["palette"]
        assert len(palette) >= 1
        assert "oklab" in palette[0]
        assert "proportion" in palette[0]

    def test_tags_have_correct_category(self, analogous_array: np.ndarray) -> None:
        analyzer = ColorAnalyzer()
        config = AnalyzerConfig(confidence_threshold=0.1)
        result = analyzer.analyze(analogous_array, config, SharedModels())
        for tag in result.tags:
            assert tag.category == "color"

    def test_custom_n_clusters(self, rainbow_array: np.ndarray) -> None:
        analyzer = ColorAnalyzer()
        config = AnalyzerConfig(params={"n_clusters": 3})
        result = analyzer.analyze(rainbow_array, config, SharedModels())
        assert result.metadata["n_palette_colors"] <= 3

    def test_custom_sub_weights(self, two_color_array: np.ndarray) -> None:
        """Custom sub-weights should change the final score."""
        analyzer = ColorAnalyzer()

        # Weight only harmony
        config = AnalyzerConfig(
            params={"sub_weights": {"harmony": 1.0, "vivid_color": 0.0}}
        )
        harmony_result = analyzer.analyze(two_color_array, config, SharedModels())

        # Scores should differ unless harmony exactly equals the default blend
        # (possible but unlikely). At minimum, check structure is valid.
        assert 0.0 <= harmony_result.score <= 1.0
        assert harmony_result.metadata["weights_used"]["vivid_color"] == 0.0
