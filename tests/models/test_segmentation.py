"""Tests for the anime segmentation model wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from loupe.models.segmentation import AnimeSegmentation


class TestAnimeSegmentationPreprocess:
    """Tests for segmentation preprocessing logic."""

    def test_not_loaded_raises(self) -> None:
        model = AnimeSegmentation(gpu=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_preprocess_landscape(self) -> None:
        model = AnimeSegmentation(gpu=False, input_size=256)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, ph, pw, h, w = model._preprocess(image)

        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.dtype == np.float32
        # Landscape: width maps to 256, height proportional
        assert w == 256
        assert h == 128
        assert pw == 0
        assert ph == 256 - 128

    def test_preprocess_portrait(self) -> None:
        model = AnimeSegmentation(gpu=False, input_size=256)
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        tensor, _ph, _pw, h, w = model._preprocess(image)

        assert tensor.shape == (1, 3, 256, 256)
        assert h == 256
        assert w == 128

    def test_preprocess_square(self) -> None:
        model = AnimeSegmentation(gpu=False, input_size=256)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, ph, pw, h, w = model._preprocess(image)

        assert tensor.shape == (1, 3, 256, 256)
        assert h == 256
        assert w == 256
        assert ph == 0
        assert pw == 0

    def test_preprocess_normalizes_to_01(self) -> None:
        model = AnimeSegmentation(gpu=False, input_size=64)
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        tensor, _, _, _, _ = model._preprocess(image)

        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0


class TestAnimeSegmentationPredict:
    """Tests for segmentation prediction with mocked ONNX session."""

    @patch("loupe.models.segmentation.create_onnx_session")
    @patch("loupe.models.segmentation.download_model")
    def test_predict_output_shape(
        self, mock_download: MagicMock, mock_create: MagicMock
    ) -> None:
        """Verify prediction returns mask with original image dimensions."""
        input_size = 256
        mock_session = MagicMock()
        # Simulate ISNet output: (1, 1, input_size, input_size) sigmoid mask
        mock_session.run.return_value = [
            np.ones((1, 1, input_size, input_size), dtype=np.float32) * 0.8
        ]
        mock_input = MagicMock()
        mock_input.name = "img"
        mock_output = MagicMock()
        mock_output.name = "mask"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_create.return_value = mock_session

        model = AnimeSegmentation(gpu=False, input_size=input_size)
        model.load()

        image = np.zeros((100, 150, 3), dtype=np.uint8)
        mask = model.predict(image)

        assert mask.shape == (100, 150)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    @patch("loupe.models.segmentation.create_onnx_session")
    @patch("loupe.models.segmentation.download_model")
    def test_predict_binary_mask(
        self, mock_download: MagicMock, mock_create: MagicMock
    ) -> None:
        """Verify that a clear foreground/background produces distinct values."""
        input_size = 64
        # Create mock output: left half = foreground, right half = background
        mock_mask = np.zeros((1, 1, input_size, input_size), dtype=np.float32)
        mock_mask[:, :, :, : input_size // 2] = 1.0
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_mask]
        mock_input = MagicMock()
        mock_input.name = "img"
        mock_output = MagicMock()
        mock_output.name = "mask"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_create.return_value = mock_session

        model = AnimeSegmentation(gpu=False, input_size=input_size)
        model.load()

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = model.predict(image)

        assert mask.shape == (64, 64)
        # Left side should be high, right side low
        assert mask[:, :16].mean() > mask[:, 48:].mean()

    def test_is_loaded_property(self) -> None:
        model = AnimeSegmentation(gpu=False)
        assert not model.is_loaded

    @patch("loupe.models.segmentation.create_onnx_session")
    @patch("loupe.models.segmentation.download_model")
    def test_is_loaded_after_load(
        self, mock_download: MagicMock, mock_create: MagicMock
    ) -> None:
        mock_create.return_value = MagicMock()
        model = AnimeSegmentation(gpu=False)
        model.load()
        assert model.is_loaded

    @patch("loupe.models.segmentation.download_model")
    def test_download_static(self, mock_download: MagicMock) -> None:
        AnimeSegmentation.download()
        mock_download.assert_called_once()
