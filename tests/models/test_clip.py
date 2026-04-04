"""Tests for the CLIP model wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from loupe.models.clip import CLIPModel


class TestCLIPModel:
    """Tests for the CLIP model wrapper."""

    def test_not_loaded_raises_image(self) -> None:
        model = CLIPModel(gpu=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.get_image_embedding(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_not_loaded_raises_text(self) -> None:
        model = CLIPModel(gpu=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.get_text_embeddings(["test"])

    def test_not_loaded_raises_classify(self) -> None:
        model = CLIPModel(gpu=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.zero_shot_classify(
                np.zeros((100, 100, 3), dtype=np.uint8), ["a", "b"]
            )

    def test_is_loaded_property(self) -> None:
        model = CLIPModel(gpu=False)
        assert not model.is_loaded


class TestCLIPModelPredict:
    """Tests for CLIP model predictions with mocked open_clip."""

    def _setup_mock_clip(self) -> tuple[CLIPModel, MagicMock]:
        """Create a CLIPModel with mocked internals."""
        model = CLIPModel(gpu=False)
        mock_clip_model = MagicMock()

        # Mock encode_image: return a random normalized embedding
        emb = torch.randn(1, 768)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mock_clip_model.encode_image.return_value = emb
        mock_clip_model.to.return_value = mock_clip_model
        mock_clip_model.eval.return_value = mock_clip_model
        mock_clip_model.half.return_value = mock_clip_model

        # Mock encode_text
        text_embs = torch.randn(3, 768)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        mock_clip_model.encode_text.return_value = text_embs

        model._model = mock_clip_model
        model._device = torch.device("cpu")

        # Mock preprocess
        mock_preprocess = MagicMock()
        mock_preprocess.return_value = torch.zeros(3, 224, 224)
        model._preprocess = mock_preprocess

        # Mock tokenizer
        model._tokenizer = MagicMock(return_value=torch.zeros(3, 77, dtype=torch.long))

        return model, mock_clip_model

    def test_image_embedding_shape(self) -> None:
        model, _ = self._setup_mock_clip()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        embedding = model.get_image_embedding(image)

        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32

    def test_image_embedding_normalized(self) -> None:
        model, _ = self._setup_mock_clip()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        embedding = model.get_image_embedding(image)

        # L2 norm should be ~1.0
        norm = np.linalg.norm(embedding)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_text_embeddings_shape(self) -> None:
        model, _ = self._setup_mock_clip()
        texts = ["anime style", "realistic", "abstract"]
        embeddings = model.get_text_embeddings(texts)

        assert embeddings.shape == (3, 768)
        assert embeddings.dtype == np.float32

    def test_text_embeddings_normalized(self) -> None:
        model, _ = self._setup_mock_clip()
        texts = ["test1", "test2", "test3"]
        embeddings = model.get_text_embeddings(texts)

        for i in range(3):
            norm = np.linalg.norm(embeddings[i])
            assert norm == pytest.approx(1.0, abs=0.01)

    def test_zero_shot_classify_probabilities(self) -> None:
        model, _ = self._setup_mock_clip()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        labels = ["style_a", "style_b", "style_c"]
        probs = model.zero_shot_classify(image, labels)

        assert set(probs.keys()) == set(labels)
        # Probabilities should sum to 1
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)
        # All probabilities should be non-negative
        assert all(p >= 0.0 for p in probs.values())
