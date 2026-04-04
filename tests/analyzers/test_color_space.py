"""Tests for OkLab/OkLCh color space conversions."""

import numpy as np
import pytest

from loupe.analyzers._color_space import (
    linear_to_oklab,
    oklab_to_oklch,
    srgb_to_linear,
    srgb_uint8_to_oklab,
    srgb_uint8_to_oklch,
)


class TestSrgbToLinear:
    def test_black(self) -> None:
        result = srgb_to_linear(np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32))
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_white(self) -> None:
        result = srgb_to_linear(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32))
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_mid_gray(self) -> None:
        """sRGB 0.5 should decode to ~0.214 in linear."""
        result = srgb_to_linear(np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32))
        assert result.shape == (1, 1, 3)
        np.testing.assert_allclose(result[0, 0, 0], 0.214, atol=0.01)

    def test_low_values_linear_region(self) -> None:
        """Values below 0.04045 use linear mapping."""
        result = srgb_to_linear(np.array([[[0.02, 0.02, 0.02]]], dtype=np.float32))
        expected = 0.02 / 12.92
        np.testing.assert_allclose(result[0, 0, 0], expected, atol=1e-6)

    def test_preserves_shape(self) -> None:
        arr = np.random.default_rng(0).random((50, 30, 3)).astype(np.float32)
        result = srgb_to_linear(arr)
        assert result.shape == (50, 30, 3)
        assert result.dtype == np.float32


class TestLinearToOklab:
    def test_black(self) -> None:
        linear = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        lab = linear_to_oklab(linear)
        # Black should have L≈0, a≈0, b≈0
        np.testing.assert_allclose(lab[0, 0], [0.0, 0.0, 0.0], atol=1e-5)

    def test_white(self) -> None:
        linear = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        lab = linear_to_oklab(linear)
        # White should have L≈1, a≈0, b≈0
        np.testing.assert_allclose(lab[0, 0, 0], 1.0, atol=0.01)
        np.testing.assert_allclose(lab[0, 0, 1], 0.0, atol=0.01)
        np.testing.assert_allclose(lab[0, 0, 2], 0.0, atol=0.01)

    def test_pure_red_has_positive_a(self) -> None:
        """Pure red in OkLab should have positive a (red-green axis)."""
        linear = srgb_to_linear(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
        lab = linear_to_oklab(linear)
        assert lab[0, 0, 1] > 0  # a > 0 for red

    def test_lightness_ordering(self) -> None:
        """Darker colors should have lower L values."""
        dark = srgb_to_linear(np.array([[[0.2, 0.2, 0.2]]], dtype=np.float32))
        light = srgb_to_linear(np.array([[[0.8, 0.8, 0.8]]], dtype=np.float32))
        dark_lab = linear_to_oklab(dark)
        light_lab = linear_to_oklab(light)
        assert dark_lab[0, 0, 0] < light_lab[0, 0, 0]


class TestOklabToOklch:
    def test_achromatic(self) -> None:
        """Achromatic color (a=0, b=0) should have C=0."""
        lab = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32)
        lch = oklab_to_oklch(lab)
        assert lch[0, 0, 0] == pytest.approx(0.5)
        assert lch[0, 0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_positive_a_hue(self) -> None:
        """Positive a, zero b → hue ≈ 0°."""
        lab = np.array([[[0.5, 0.1, 0.0]]], dtype=np.float32)
        lch = oklab_to_oklch(lab)
        assert lch[0, 0, 2] == pytest.approx(0.0, abs=0.1)

    def test_positive_b_hue(self) -> None:
        """Zero a, positive b → hue ≈ 90°."""
        lab = np.array([[[0.5, 0.0, 0.1]]], dtype=np.float32)
        lch = oklab_to_oklch(lab)
        assert lch[0, 0, 2] == pytest.approx(90.0, abs=0.1)

    def test_chroma_magnitude(self) -> None:
        lab = np.array([[[0.5, 0.3, 0.4]]], dtype=np.float32)
        lch = oklab_to_oklch(lab)
        expected_c = np.sqrt(0.3**2 + 0.4**2)
        assert lch[0, 0, 1] == pytest.approx(expected_c, abs=1e-5)

    def test_hue_range(self) -> None:
        """All hue values should be in [0, 360)."""
        lab = np.array(
            [[[0.5, -0.1, -0.1], [0.5, 0.1, -0.1], [0.5, -0.1, 0.1]]],
            dtype=np.float32,
        )
        lch = oklab_to_oklch(lab)
        assert np.all(lch[..., 2] >= 0.0)
        assert np.all(lch[..., 2] < 360.0)


class TestConvenienceFunctions:
    def test_srgb_uint8_to_oklab(self) -> None:
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        lab = srgb_uint8_to_oklab(image)
        assert lab.shape == (10, 10, 3)
        assert lab.dtype == np.float32
        # Mid-gray should have L around 0.5-0.6
        assert 0.4 < lab[0, 0, 0] < 0.7

    def test_srgb_uint8_to_oklch(self) -> None:
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        lch = srgb_uint8_to_oklch(image)
        assert lch.shape == (10, 10, 3)
        assert lch.dtype == np.float32

    def test_pure_colors_distinguishable(self) -> None:
        """Red, green, blue should have distinct hues in OkLCh."""
        red = np.full((5, 5, 3), [255, 0, 0], dtype=np.uint8)
        green = np.full((5, 5, 3), [0, 255, 0], dtype=np.uint8)
        blue = np.full((5, 5, 3), [0, 0, 255], dtype=np.uint8)

        red_lch = srgb_uint8_to_oklch(red)
        green_lch = srgb_uint8_to_oklch(green)
        blue_lch = srgb_uint8_to_oklch(blue)

        # All should have nonzero chroma
        assert red_lch[0, 0, 1] > 0.05
        assert green_lch[0, 0, 1] > 0.05
        assert blue_lch[0, 0, 1] > 0.05

        # Hues should be distinct (at least 60° apart)
        hues = [red_lch[0, 0, 2], green_lch[0, 0, 2], blue_lch[0, 0, 2]]
        for i in range(3):
            for j in range(i + 1, 3):
                diff = abs(hues[i] - hues[j])
                diff = min(diff, 360 - diff)
                assert diff > 60, f"Hues {hues[i]:.1f} and {hues[j]:.1f} too close"
