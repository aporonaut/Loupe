"""Anime aesthetic scorer — quality estimation via deepghs SwinV2.

Uses the deepghs/anime_aesthetic ONNX model to classify anime images
into 7 quality tiers (worst → masterpiece) and produce a continuous
0.0-1.0 aesthetic quality score via probability-weighted sum.

This is a quality proxy, not a pure aesthetic measure — it reflects
the model's training on Danbooru quality ratings, which correlate with
but do not perfectly capture aesthetic merit.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from loupe.models.onnx_utils import create_onnx_session, download_model

logger = logging.getLogger(__name__)

REPO_ID = "deepghs/anime_aesthetic"
VARIANT = "swinv2pv3_v0_448_ls0.2"
ONNX_PATH = f"{VARIANT}/model.onnx"
INPUT_SIZE = 448

# Quality tier labels in output order (index 0 = worst, index 6 = masterpiece)
QUALITY_LABELS: list[str] = [
    "worst",
    "low",
    "normal",
    "good",
    "great",
    "best",
    "masterpiece",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class AnimeAestheticScorer:
    """Anime aesthetic quality scorer via deepghs SwinV2 (ONNX).

    Parameters
    ----------
    gpu : bool
        Use CUDA if available.
    """

    def __init__(self, *, gpu: bool = True) -> None:
        self._gpu = gpu
        self._session: Any | None = None

    def load(self) -> None:
        """Download and load the ONNX model."""
        model_path = download_model(REPO_ID, ONNX_PATH, local_only=True)
        self._session = create_onnx_session(model_path, gpu=self._gpu)
        logger.info("Anime aesthetic scorer loaded")

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._session is not None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for the aesthetic scorer.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        np.ndarray
            Float32 tensor (1, 3, 448, 448) normalized to [-1, 1].
        """
        resized = cv2.resize(
            image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        normalized = resized.astype(np.float32) / 255.0
        # Normalize to [-1, 1]: (x - 0.5) / 0.5
        normalized = (normalized - 0.5) / 0.5
        # CHW + batch
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, :]
        return tensor

    def predict(self, image: np.ndarray) -> tuple[float, str, dict[str, float]]:
        """Score an image's aesthetic quality.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        tuple[float, str, dict[str, float]]
            (score, tier, tier_probabilities) where:
            - score: 0.0-1.0 continuous quality score
            - tier: best matching quality label
            - tier_probabilities: per-tier probability dict
        """
        if self._session is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        tensor = self._preprocess(image)

        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        raw_output = self._session.run([output_name], {input_name: tensor})[0][0]

        # Apply softmax to get probabilities
        probs = _softmax(raw_output)

        # Build tier probability dict
        tier_probs: dict[str, float] = {}
        for i, label in enumerate(QUALITY_LABELS):
            tier_probs[label] = float(probs[i])

        # Best tier by argmax
        best_tier = QUALITY_LABELS[int(np.argmax(probs))]

        # Continuous score: probability-weighted sum normalized to [0, 1]
        # Index 0 (worst) = 0.0, index 6 (masterpiece) = 1.0
        score = float(sum(probs[i] * i for i in range(len(QUALITY_LABELS))) / 6.0)
        score = max(0.0, min(1.0, score))

        return score, best_tier, tier_probs

    @staticmethod
    def download() -> None:
        """Pre-download model files without loading."""
        download_model(REPO_ID, ONNX_PATH)
        logger.info("Anime aesthetic scorer model downloaded")
