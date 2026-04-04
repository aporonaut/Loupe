"""Tests for the anime aesthetic scorer wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from loupe.models.aesthetic import QUALITY_LABELS, AnimeAestheticScorer, _softmax


class TestSoftmax:
    """Tests for the softmax utility."""

    def test_softmax_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = _softmax(x)
        assert result.sum() == pytest.approx(1.0)

    def test_softmax_preserves_order(self) -> None:
        x = np.array([1.0, 3.0, 2.0])
        result = _softmax(x)
        assert result[1] > result[2] > result[0]

    def test_softmax_uniform(self) -> None:
        x = np.array([0.0, 0.0, 0.0])
        result = _softmax(x)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3])

    def test_softmax_numerical_stability(self) -> None:
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _softmax(x)
        assert not np.any(np.isnan(result))
        assert result.sum() == pytest.approx(1.0)


class TestAnimeAestheticScorer:
    """Tests for the aesthetic scorer."""

    def test_not_loaded_raises(self) -> None:
        scorer = AnimeAestheticScorer(gpu=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_is_loaded_property(self) -> None:
        scorer = AnimeAestheticScorer(gpu=False)
        assert not scorer.is_loaded

    def test_preprocess_shape(self) -> None:
        scorer = AnimeAestheticScorer(gpu=False)
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        tensor = scorer._preprocess(image)
        assert tensor.shape == (1, 3, 448, 448)
        assert tensor.dtype == np.float32

    def test_preprocess_normalization_range(self) -> None:
        scorer = AnimeAestheticScorer(gpu=False)
        # All-white image
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        tensor = scorer._preprocess(image)
        # After (x/255 - 0.5) / 0.5: 255/255 = 1.0 → (1.0 - 0.5) / 0.5 = 1.0
        assert tensor.max() == pytest.approx(1.0)

        # All-black image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor = scorer._preprocess(image)
        # 0/255 = 0.0 → (0.0 - 0.5) / 0.5 = -1.0
        assert tensor.min() == pytest.approx(-1.0)

    @patch("loupe.models.aesthetic.create_onnx_session")
    @patch("loupe.models.aesthetic.download_model")
    def test_predict_masterpiece(
        self, mock_download: MagicMock, mock_create: MagicMock
    ) -> None:
        """Masterpiece-heavy logits should produce high score."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        # Strong masterpiece logits (index 6)
        mock_session.run.return_value = [
            np.array([[-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 10.0]])
        ]
        mock_create.return_value = mock_session

        scorer = AnimeAestheticScorer(gpu=False)
        scorer.load()

        score, tier, tier_probs = scorer.predict(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        assert score > 0.9
        assert tier == "masterpiece"
        assert tier_probs["masterpiece"] > 0.9

    @patch("loupe.models.aesthetic.create_onnx_session")
    @patch("loupe.models.aesthetic.download_model")
    def test_predict_worst(
        self, mock_download: MagicMock, mock_create: MagicMock
    ) -> None:
        """Worst-heavy logits should produce low score."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        # Strong worst logits (index 0)
        mock_session.run.return_value = [
            np.array([[10.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
        ]
        mock_create.return_value = mock_session

        scorer = AnimeAestheticScorer(gpu=False)
        scorer.load()

        score, tier, tier_probs = scorer.predict(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        assert score < 0.1
        assert tier == "worst"
        assert tier_probs["worst"] > 0.9

    @patch("loupe.models.aesthetic.create_onnx_session")
    @patch("loupe.models.aesthetic.download_model")
    def test_predict_score_range(
        self, mock_download: MagicMock, mock_create: MagicMock
    ) -> None:
        """Score should always be in [0.0, 1.0]."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        # Uniform logits
        mock_session.run.return_value = [
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ]
        mock_create.return_value = mock_session

        scorer = AnimeAestheticScorer(gpu=False)
        scorer.load()

        score, _tier, tier_probs = scorer.predict(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        assert 0.0 <= score <= 1.0
        # Uniform probs → weighted mean index = 3.0, score = 3.0/6.0 = 0.5
        assert score == pytest.approx(0.5, abs=0.01)
        assert len(tier_probs) == len(QUALITY_LABELS)
        assert sum(tier_probs.values()) == pytest.approx(1.0)

    @patch("loupe.models.aesthetic.download_model")
    def test_download_static(self, mock_download: MagicMock) -> None:
        AnimeAestheticScorer.download()
        mock_download.assert_called_once()
