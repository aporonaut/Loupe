# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Anime segmentation model — character/background mask via ISNet-IS.

Uses the skytnt/anime-seg ONNX model (ISNet-IS architecture) to produce
per-pixel character masks. The model outputs a sigmoid-activated mask
where 1.0 = foreground (character) and 0.0 = background.

Limitations:
- Designed for anime/illustration content; photographic content
  produces unreliable masks.
- Multi-character scenes may merge overlapping characters into
  a single foreground region.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from loupe.models.onnx_utils import create_onnx_session, download_model

logger = logging.getLogger(__name__)

REPO_ID = "skytnt/anime-seg"
ONNX_FILENAME = "isnetis.onnx"
DEFAULT_INPUT_SIZE = 1024


class AnimeSegmentation:
    """Anime character segmentation via ISNet-IS (ONNX).

    Parameters
    ----------
    gpu : bool
        Use CUDA if available.
    input_size : int
        Model input resolution. Higher values produce finer masks
        at the cost of more VRAM and compute.
    """

    def __init__(
        self, *, gpu: bool = True, input_size: int = DEFAULT_INPUT_SIZE
    ) -> None:
        self._gpu = gpu
        self._input_size = input_size
        self._session: Any | None = None

    def load(self) -> None:
        """Download and load the ONNX model."""
        model_path = download_model(REPO_ID, ONNX_FILENAME, local_only=True)
        self._session = create_onnx_session(model_path, gpu=self._gpu)
        logger.info("Anime segmentation model loaded (input_size=%d)", self._input_size)

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._session is not None

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
        """Preprocess image for ISNet-IS inference.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        tuple[np.ndarray, int, int, int, int]
            (input_tensor, pad_h, pad_w, resized_h, resized_w)
        """
        s = self._input_size
        h0, w0 = image.shape[:2]

        # Resize maintaining aspect ratio to fit within s x s
        if h0 > w0:
            h, w = s, int(s * w0 / h0)
        else:
            h, w = int(s * h0 / w0), s

        # Ensure non-zero dimensions
        h = max(h, 1)
        w = max(w, 1)

        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Zero-pad to s x s, centered
        ph = s - h
        pw = s - w
        canvas = np.zeros((s, s, 3), dtype=np.float32)
        y_off = ph // 2
        x_off = pw // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = normalized

        # CHW format + batch dimension
        tensor = np.transpose(canvas, (2, 0, 1))[np.newaxis, :]
        return tensor, ph, pw, h, w

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Produce a character segmentation mask.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        np.ndarray
            Float32 mask (H, W) with values in [0.0, 1.0].
            1.0 = foreground (character), 0.0 = background.
        """
        if self._session is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        h0, w0 = image.shape[:2]
        tensor, ph, pw, h, w = self._preprocess(image)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        result = self._session.run([output_name], {input_name: tensor})

        # Post-process: remove batch and channel dims
        mask: np.ndarray = np.asarray(result[0][0, 0])  # (s, s)

        # Crop out padding
        y_off = ph // 2
        x_off = pw // 2
        mask = mask[y_off : y_off + h, x_off : x_off + w]

        # Resize back to original dimensions
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_LINEAR)

        # Clamp to [0, 1]
        return np.clip(mask, 0.0, 1.0)  # type: ignore[return-value]

    @staticmethod
    def download() -> None:
        """Pre-download the model file without loading it."""
        download_model(REPO_ID, ONNX_FILENAME)
        logger.info("Anime segmentation model downloaded")
