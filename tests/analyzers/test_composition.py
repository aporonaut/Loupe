"""Tests for the composition analyzer."""

import numpy as np
import pytest

from loupe.analyzers.base import AnalyzerConfig, SharedModels
from loupe.analyzers.composition import (
    CompositionAnalyzer,
    _combine_scores,
    compute_saliency,
    measure_depth_layering,
    measure_diagonal_dominance,
    measure_framing,
    measure_leading_lines,
    measure_negative_space,
    measure_rule_of_thirds,
    measure_symmetry,
    measure_visual_balance,
)

# -- Fixtures --


@pytest.fixture
def uniform_gray() -> np.ndarray:
    """Uniform gray 100x100 image (no edges, no structure)."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def centered_subject() -> np.ndarray:
    """Image with a bright subject centered on gray background."""
    arr = np.full((200, 200, 3), 60, dtype=np.uint8)
    # Bright square in center
    arr[70:130, 70:130] = 240
    return arr


@pytest.fixture
def thirds_subject() -> np.ndarray:
    """Image with a bright subject at the upper-left thirds intersection."""
    arr = np.full((300, 300, 3), 60, dtype=np.uint8)
    # Bright square near (100, 100) — the upper-left power point
    arr[80:120, 80:120] = 240
    return arr


@pytest.fixture
def symmetric_image() -> np.ndarray:
    """Horizontally symmetric image with edge structure."""
    arr = np.full((100, 200, 3), 60, dtype=np.uint8)
    # Symmetric vertical bars
    arr[:, 30:40] = 240
    arr[:, 160:170] = 240
    # Center line
    arr[:, 95:105] = 180
    return arr


@pytest.fixture
def asymmetric_image() -> np.ndarray:
    """Asymmetric image — subject only on the left."""
    arr = np.full((100, 200, 3), 60, dtype=np.uint8)
    arr[:, 10:50] = 240
    return arr


@pytest.fixture
def diagonal_lines_image() -> np.ndarray:
    """Image with strong diagonal lines drawn on it."""
    import cv2

    arr = np.full((200, 200, 3), 60, dtype=np.uint8)
    # Draw diagonal lines at ~45 degrees
    cv2.line(arr, (20, 20), (180, 180), (240, 240, 240), 2)
    cv2.line(arr, (20, 180), (180, 20), (240, 240, 240), 2)
    cv2.line(arr, (50, 10), (190, 150), (200, 200, 200), 2)
    return arr


@pytest.fixture
def open_image() -> np.ndarray:
    """Image with significant negative space (mostly empty)."""
    arr = np.full((200, 200, 3), 60, dtype=np.uint8)
    # Small subject in one corner — most of image is uniform
    arr[10:30, 10:30] = 240
    return arr


@pytest.fixture
def cluttered_image() -> np.ndarray:
    """Cluttered image with random high-frequency noise (lots of edges)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)


@pytest.fixture
def framed_image() -> np.ndarray:
    """Image with border edges framing a clean center."""
    arr = np.full((200, 200, 3), 80, dtype=np.uint8)
    # Draw rectangle border in outer region
    arr[:15, :] = 240  # top bar
    arr[185:, :] = 240  # bottom bar
    arr[:, :15] = 240  # left bar
    arr[:, 185:] = 240  # right bar
    # Center is uniform — no edges
    arr[40:160, 40:160] = 80
    return arr


@pytest.fixture
def depth_image() -> np.ndarray:
    """Image with different sharpness across horizontal strips."""
    arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Top strip: blurry (uniform)
    arr[:80, :] = 100
    # Bottom strip: sharp edges (high-frequency checkerboard)
    for y in range(120, 200):
        for x in range(200):
            if (y + x) % 4 < 2:
                arr[y, x] = 220
            else:
                arr[y, x] = 40
    return arr


# -- Saliency Tests --


class TestComputeSaliency:
    def test_output_shape(self, centered_subject: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        assert saliency.shape == gray.shape
        assert saliency.dtype == np.float32

    def test_value_range(self, centered_subject: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        assert saliency.min() >= 0.0
        assert saliency.max() <= 1.0

    def test_centered_subject_peak_near_center(
        self, centered_subject: np.ndarray
    ) -> None:
        """Saliency should peak near the centered subject."""
        import cv2

        gray = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        h, w = saliency.shape
        center_region = saliency[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        border_region_mean = (saliency.sum() - center_region.sum()) / (
            saliency.size - center_region.size
        )
        assert center_region.mean() > border_region_mean

    def test_uniform_image_low_saliency(self, uniform_gray: np.ndarray) -> None:
        """Uniform image should have near-zero saliency everywhere."""
        import cv2

        gray = cv2.cvtColor(uniform_gray, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        # No edges -> density is zero -> saliency is zero
        assert saliency.max() < 0.01


# -- Rule of Thirds Tests --


class TestRuleOfThirds:
    def test_thirds_subject_high(self, thirds_subject: np.ndarray) -> None:
        """Subject at thirds intersection should score relatively high."""
        import cv2

        gray = cv2.cvtColor(thirds_subject, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        score = measure_rule_of_thirds(saliency)
        assert score > 0.3

    def test_score_range(self, centered_subject: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        score = measure_rule_of_thirds(saliency)
        assert 0.0 <= score <= 1.0

    def test_empty_saliency_returns_zero(self) -> None:
        saliency = np.zeros((100, 100), dtype=np.float32)
        assert measure_rule_of_thirds(saliency) == 0.0

    def test_tiny_image(self) -> None:
        saliency = np.ones((2, 2), dtype=np.float32)
        assert measure_rule_of_thirds(saliency) == 0.0


# -- Visual Balance Tests --


class TestVisualBalance:
    def test_centered_subject_balanced(self, centered_subject: np.ndarray) -> None:
        """Centered subject should produce high balance score."""
        import cv2

        gray = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        score = measure_visual_balance(saliency)
        assert score > 0.6

    def test_asymmetric_lower_balance(
        self,
        centered_subject: np.ndarray,
        asymmetric_image: np.ndarray,
    ) -> None:
        """Asymmetric image should have lower balance than centered."""
        import cv2

        gray_centered = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        gray_asym = cv2.cvtColor(asymmetric_image, cv2.COLOR_RGB2GRAY)
        sal_centered = compute_saliency(gray_centered)
        sal_asym = compute_saliency(gray_asym)
        assert measure_visual_balance(sal_centered) > measure_visual_balance(sal_asym)

    def test_empty_saliency_neutral(self) -> None:
        saliency = np.zeros((100, 100), dtype=np.float32)
        assert measure_visual_balance(saliency) == 0.5

    def test_score_range(self, cluttered_image: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(cluttered_image, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        score = measure_visual_balance(saliency)
        assert 0.0 <= score <= 1.0


# -- Symmetry Tests --


class TestSymmetry:
    def test_symmetric_image_high(self, symmetric_image: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(symmetric_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        score = measure_symmetry(edges)
        assert score > 0.3

    def test_asymmetric_image_lower(
        self,
        symmetric_image: np.ndarray,
        asymmetric_image: np.ndarray,
    ) -> None:
        import cv2

        gray_sym = cv2.cvtColor(symmetric_image, cv2.COLOR_RGB2GRAY)
        gray_asym = cv2.cvtColor(asymmetric_image, cv2.COLOR_RGB2GRAY)
        edges_sym = cv2.Canny(gray_sym, 50, 150)
        edges_asym = cv2.Canny(gray_asym, 50, 150)
        assert measure_symmetry(edges_sym) > measure_symmetry(edges_asym)

    def test_no_edges_returns_zero(self) -> None:
        edges = np.zeros((100, 100), dtype=np.uint8)
        assert measure_symmetry(edges) == 0.0

    def test_score_range(self, cluttered_image: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(cluttered_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        score = measure_symmetry(edges)
        assert 0.0 <= score <= 1.0


# -- Depth Layering Tests --


class TestDepthLayering:
    def test_depth_image_higher(self, depth_image: np.ndarray) -> None:
        """Image with varying sharpness should score higher than uniform."""
        import cv2

        gray_depth = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
        gray_uniform = np.full((200, 200), 128, dtype=np.uint8)
        assert measure_depth_layering(gray_depth) > measure_depth_layering(gray_uniform)

    def test_uniform_image_low(self) -> None:
        gray = np.full((200, 200), 128, dtype=np.uint8)
        assert measure_depth_layering(gray) == 0.0

    def test_score_range(self, cluttered_image: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(cluttered_image, cv2.COLOR_RGB2GRAY)
        score = measure_depth_layering(gray)
        assert 0.0 <= score <= 1.0

    def test_tiny_image(self) -> None:
        gray = np.full((5, 5), 128, dtype=np.uint8)
        assert measure_depth_layering(gray) == 0.0


# -- Leading Lines Tests --


class TestLeadingLines:
    def test_no_lines_returns_zero(self) -> None:
        saliency = np.ones((100, 100), dtype=np.float32) * 0.5
        lines = np.empty((0, 4), dtype=np.float32)
        assert measure_leading_lines(lines, saliency) == 0.0

    def test_score_range(self, diagonal_lines_image: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(diagonal_lines_image, cv2.COLOR_RGB2GRAY)
        saliency = compute_saliency(gray)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        result = lsd.detect(gray)
        lines_raw = result[0]
        if lines_raw is not None and len(lines_raw) > 0:
            lines = lines_raw.reshape(-1, 4).astype(np.float32)
            score = measure_leading_lines(lines, saliency)
            assert 0.0 <= score <= 1.0


# -- Diagonal Dominance Tests --


class TestDiagonalDominance:
    def test_no_lines_returns_zero(self) -> None:
        lines = np.empty((0, 4), dtype=np.float32)
        assert measure_diagonal_dominance(lines) == 0.0

    def test_diagonal_lines_high(self) -> None:
        """Lines at 45 degrees should produce high diagonal score."""
        # Create lines at 45 degrees
        lines = np.array(
            [
                [0, 0, 100, 100],
                [0, 100, 100, 0],
                [10, 10, 90, 90],
            ],
            dtype=np.float32,
        )
        score = measure_diagonal_dominance(lines)
        assert score > 0.5

    def test_horizontal_lines_low(self) -> None:
        """Purely horizontal lines should produce low diagonal score."""
        lines = np.array(
            [
                [0, 50, 200, 50],
                [0, 100, 200, 100],
                [0, 150, 200, 150],
            ],
            dtype=np.float32,
        )
        score = measure_diagonal_dominance(lines)
        assert score < 0.1

    def test_vertical_lines_low(self) -> None:
        """Purely vertical lines should produce low diagonal score."""
        lines = np.array(
            [
                [50, 0, 50, 200],
                [100, 0, 100, 200],
            ],
            dtype=np.float32,
        )
        score = measure_diagonal_dominance(lines)
        assert score < 0.1


# -- Negative Space Tests --


class TestNegativeSpace:
    def test_balanced_negative_space_scores_well(self) -> None:
        """Image with balanced mix of detail and empty space should score well."""
        import cv2

        # Create image with roughly 40-50% busy blocks and 50-60% empty blocks
        arr = np.full((200, 200, 3), 60, dtype=np.uint8)
        # Add structure in right half (creates many edge blocks)
        for y in range(0, 200, 10):
            arr[y : y + 2, 100:] = 240
        for x in range(100, 200, 15):
            arr[:, x : x + 2] = 240

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        score = measure_negative_space(edges)
        assert score > 0.4

    def test_cluttered_vs_balanced(self, cluttered_image: np.ndarray) -> None:
        """Cluttered image should score differently from balanced composition."""
        import cv2

        # Balanced: structure on one side, empty on other
        arr = np.full((200, 200, 3), 60, dtype=np.uint8)
        for y in range(0, 200, 10):
            arr[y : y + 2, 100:] = 240
        for x in range(100, 200, 15):
            arr[:, x : x + 2] = 240

        gray_balanced = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray_clutter = cv2.cvtColor(cluttered_image, cv2.COLOR_RGB2GRAY)
        edges_balanced = cv2.Canny(gray_balanced, 50, 150)
        edges_clutter = cv2.Canny(gray_clutter, 50, 150)
        score_balanced = measure_negative_space(edges_balanced)
        score_clutter = measure_negative_space(edges_clutter)
        # Balanced composition should beat cluttered noise
        assert score_balanced > score_clutter

    def test_score_range(self, centered_subject: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        score = measure_negative_space(edges)
        assert 0.0 <= score <= 1.0

    def test_tiny_image(self) -> None:
        edges = np.zeros((8, 8), dtype=np.uint8)
        score = measure_negative_space(edges)
        assert 0.0 <= score <= 1.0


# -- Framing Tests --


class TestFraming:
    def test_framed_image_scores_higher(
        self,
        framed_image: np.ndarray,
        centered_subject: np.ndarray,
    ) -> None:
        """Framed image should score higher than centered subject (no frame)."""
        import cv2

        gray_framed = cv2.cvtColor(framed_image, cv2.COLOR_RGB2GRAY)
        gray_centered = cv2.cvtColor(centered_subject, cv2.COLOR_RGB2GRAY)
        edges_framed = cv2.Canny(gray_framed, 50, 150)
        edges_centered = cv2.Canny(gray_centered, 50, 150)
        assert measure_framing(edges_framed) > measure_framing(edges_centered)

    def test_no_edges_returns_zero(self) -> None:
        edges = np.zeros((100, 100), dtype=np.uint8)
        assert measure_framing(edges) == 0.0

    def test_score_range(self, cluttered_image: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(cluttered_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        score = measure_framing(edges)
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

    def test_empty_returns_zero(self) -> None:
        assert _combine_scores({}, {}) == 0.0


# -- Full Analyzer Tests --


class TestCompositionAnalyzer:
    def test_satisfies_protocol(self) -> None:
        from loupe.analyzers.base import BaseAnalyzer

        analyzer: BaseAnalyzer = CompositionAnalyzer()
        assert analyzer.name == "composition"

    def test_produces_result(self, centered_subject: np.ndarray) -> None:
        analyzer = CompositionAnalyzer()
        config = AnalyzerConfig()
        shared = SharedModels()
        result = analyzer.analyze(centered_subject, config, shared)

        assert result.analyzer == "composition"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.tags, list)
        assert "sub_scores" in result.metadata
        assert "n_lines_detected" in result.metadata

    def test_sub_scores_in_metadata(self, centered_subject: np.ndarray) -> None:
        analyzer = CompositionAnalyzer()
        result = analyzer.analyze(centered_subject, AnalyzerConfig(), SharedModels())
        sub_scores = result.metadata["sub_scores"]

        expected_keys = {
            "rule_of_thirds",
            "visual_balance",
            "symmetry",
            "depth_layering",
            "leading_lines",
            "diagonal_dominance",
            "negative_space",
            "framing",
        }
        assert set(sub_scores.keys()) == expected_keys
        for value in sub_scores.values():
            assert 0.0 <= value <= 1.0

    def test_uniform_image_low_score(self, uniform_gray: np.ndarray) -> None:
        """Uniform gray image has no compositional structure."""
        analyzer = CompositionAnalyzer()
        result = analyzer.analyze(uniform_gray, AnalyzerConfig(), SharedModels())
        assert result.score < 0.5

    def test_tags_have_correct_category(self, centered_subject: np.ndarray) -> None:
        analyzer = CompositionAnalyzer()
        config = AnalyzerConfig(confidence_threshold=0.1)
        result = analyzer.analyze(centered_subject, config, SharedModels())
        for tag in result.tags:
            assert tag.category == "composition"

    def test_custom_sub_weights(self, centered_subject: np.ndarray) -> None:
        """Custom sub-weights should be reflected in metadata."""
        analyzer = CompositionAnalyzer()
        config = AnalyzerConfig(
            params={"sub_weights": {"rule_of_thirds": 0.0, "visual_balance": 1.0}}
        )
        result = analyzer.analyze(centered_subject, config, SharedModels())
        assert result.metadata["weights_used"]["rule_of_thirds"] == 0.0
        assert result.metadata["weights_used"]["visual_balance"] == 1.0

    def test_analysis_resolution_in_metadata(
        self, centered_subject: np.ndarray
    ) -> None:
        analyzer = CompositionAnalyzer()
        result = analyzer.analyze(centered_subject, AnalyzerConfig(), SharedModels())
        assert "analysis_resolution" in result.metadata
        assert len(result.metadata["analysis_resolution"]) == 2

    def test_random_noise_produces_valid_result(
        self, cluttered_image: np.ndarray
    ) -> None:
        """Random noise should still produce valid (though low quality) result."""
        analyzer = CompositionAnalyzer()
        result = analyzer.analyze(cluttered_image, AnalyzerConfig(), SharedModels())
        assert 0.0 <= result.score <= 1.0
        assert result.analyzer == "composition"

    def test_small_image(self) -> None:
        """Very small image should still produce a valid result."""
        arr = np.full((20, 20, 3), 128, dtype=np.uint8)
        arr[8:12, 8:12] = 240
        analyzer = CompositionAnalyzer()
        result = analyzer.analyze(arr, AnalyzerConfig(), SharedModels())
        assert 0.0 <= result.score <= 1.0
