"""Tests for image loading."""

from pathlib import Path

import numpy as np
from PIL import Image

from loupe.io.image import SUPPORTED_FORMATS, load_image


class TestLoadImage:
    def test_load_png(self, solid_red_image: Path) -> None:
        result = load_image(solid_red_image)
        assert result is not None
        assert result.array.shape == (100, 100, 3)
        assert result.array.dtype == np.uint8
        # Should be red
        assert result.array[50, 50, 0] == 255
        assert result.array[50, 50, 1] == 0
        assert result.metadata.width == 100
        assert result.metadata.height == 100
        assert result.metadata.format == "PNG"

    def test_load_jpeg(self, jpeg_image: Path) -> None:
        result = load_image(jpeg_image)
        assert result is not None
        assert result.array.shape == (80, 120, 3)
        assert result.metadata.format == "JPEG"

    def test_load_webp(self, tmp_path: Path) -> None:
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        path = tmp_path / "test.webp"
        img.save(path)
        result = load_image(path)
        assert result is not None
        assert result.metadata.format == "WEBP"

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "test.gif"
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), mode="RGB")
        img.save(path)
        result = load_image(path)
        assert result is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = load_image(tmp_path / "nonexistent.png")
        assert result is None

    def test_rgba_converted_to_rgb(self, tmp_path: Path) -> None:
        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        arr[:, :, 3] = 255  # full alpha
        img = Image.fromarray(arr, mode="RGBA")
        path = tmp_path / "rgba.png"
        img.save(path)
        result = load_image(path)
        assert result is not None
        assert result.array.shape == (50, 50, 3)

    def test_supported_formats_constant(self) -> None:
        assert "PNG" in SUPPORTED_FORMATS
        assert "JPEG" in SUPPORTED_FORMATS
        assert "WEBP" in SUPPORTED_FORMATS
