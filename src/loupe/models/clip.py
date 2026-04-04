"""CLIP model — image embeddings and zero-shot classification via OpenCLIP.

Uses ViT-L/14 (OpenAI pretrained) via the open_clip library for image
embedding extraction, text embedding computation, and zero-shot
classification. Used by the Style analyzer for style tagging and
aesthetic similarity scoring.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"


class CLIPModel:
    """CLIP ViT-L/14 for image embeddings and zero-shot classification.

    Parameters
    ----------
    gpu : bool
        Use CUDA if available.
    """

    def __init__(self, *, gpu: bool = True) -> None:
        self._gpu = gpu
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._device: torch.device = torch.device("cpu")

    def load(self) -> None:
        """Download and load the CLIP model."""
        import warnings

        import open_clip  # pyright: ignore[reportMissingTypeStubs]

        self._device = torch.device(
            "cuda" if self._gpu and torch.cuda.is_available() else "cpu"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="QuickGELU mismatch")
            model, _, preprocess = open_clip.create_model_and_transforms(  # pyright: ignore[reportUnknownMemberType]
                MODEL_NAME, pretrained=PRETRAINED
            )
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)  # pyright: ignore[reportUnknownMemberType]

        # Use FP16 on GPU for lower VRAM
        if self._device.type == "cuda":
            self._model = self._model.half()

        logger.info("CLIP model loaded (device=%s)", self._device)

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._model is not None

    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Compute a normalized image embedding.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        np.ndarray
            Float32 embedding vector (768,), L2-normalized.
        """
        if self._model is None or self._preprocess is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        pil_image = Image.fromarray(image)
        input_tensor: torch.Tensor = self._preprocess(pil_image).unsqueeze(0)  # type: ignore[union-attr]
        input_tensor = input_tensor.to(self._device)

        if self._device.type == "cuda":
            input_tensor = input_tensor.half()

        with torch.no_grad():
            embedding = self._model.encode_image(input_tensor)
            # L2 normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().float().numpy()[0]

    def get_text_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute normalized text embeddings.

        Parameters
        ----------
        texts : list[str]
            Text labels to embed.

        Returns
        -------
        np.ndarray
            Float32 array (N, 768), each row L2-normalized.
        """
        if self._model is None or self._tokenizer is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().float().numpy()

    def zero_shot_classify(
        self, image: np.ndarray, labels: list[str]
    ) -> dict[str, float]:
        """Zero-shot image classification against text labels.

        Computes cosine similarity between the image embedding and
        each text label embedding, then applies softmax.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).
        labels : list[str]
            Text labels to classify against.

        Returns
        -------
        dict[str, float]
            Label to probability mapping (softmax over similarities).
        """
        image_emb = self.get_image_embedding(image)
        return self.zero_shot_classify_from_embedding(image_emb, labels)

    def zero_shot_classify_from_embedding(
        self, image_embedding: np.ndarray, labels: list[str]
    ) -> dict[str, float]:
        """Zero-shot classification using a pre-computed image embedding.

        Parameters
        ----------
        image_embedding : np.ndarray
            L2-normalized image embedding (768,).
        labels : list[str]
            Text labels to classify against.

        Returns
        -------
        dict[str, float]
            Label to probability mapping (softmax over similarities).
        """
        text_embs = self.get_text_embeddings(labels)

        # Cosine similarity (already normalized)
        similarities = text_embs @ image_embedding

        # Softmax with temperature scaling (CLIP default: 100)
        logits = similarities * 100.0
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        return {label: float(prob) for label, prob in zip(labels, probs, strict=True)}

    @staticmethod
    def download() -> None:
        """Pre-download model files without loading."""
        import open_clip  # pyright: ignore[reportMissingTypeStubs]

        open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)  # pyright: ignore[reportUnknownMemberType]
        logger.info("CLIP model downloaded")
