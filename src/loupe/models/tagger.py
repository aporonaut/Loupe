# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""WD-Tagger v3 — Danbooru-style tag prediction via SwinV2.

Uses the SmilingWolf/wd-swinv2-tagger-v3 model (via timm) to predict
anime-relevant tags with sigmoid confidence scores. Outputs a dictionary
of tag names to confidence values, filtered by a configurable threshold.

The model produces 10,861 tag predictions covering composition, character
attributes, lighting, style, and scene descriptors from Danbooru taxonomy.
"""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
)
from PIL import Image

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

REPO_ID = "SmilingWolf/wd-swinv2-tagger-v3"
TIMM_MODEL_NAME = "hf-hub:SmilingWolf/wd-swinv2-tagger-v3"
TAGS_FILENAME = "selected_tags.csv"
INPUT_SIZE = 448


class WDTagger:
    """WD-Tagger v3 (SwinV2-Base) for anime tag prediction.

    Parameters
    ----------
    gpu : bool
        Use CUDA if available.
    threshold : float
        Minimum confidence to include a tag in predictions.
    """

    def __init__(self, *, gpu: bool = True, threshold: float = 0.35) -> None:
        self._gpu = gpu
        self._threshold = threshold
        self._model: Any | None = None
        self._transform: Any | None = None
        self._tag_names: list[str] = []
        self._device: torch.device = torch.device("cpu")

    def load(self) -> None:
        """Download and load the model and tag vocabulary."""
        import timm
        import timm.data  # pyright: ignore[reportAttributeAccessIssue]

        self._device = torch.device(
            "cuda" if self._gpu and torch.cuda.is_available() else "cpu"
        )

        self._model = timm.create_model(TIMM_MODEL_NAME, pretrained=True)
        self._model = self._model.to(self._device).eval()

        # Get the timm preprocessing transform
        data_cfg = timm.data.resolve_model_data_config(self._model)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportPrivateImportUsage]
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportPrivateImportUsage]

        # Load tag vocabulary
        tags_path: str = hf_hub_download(REPO_ID, TAGS_FILENAME, local_files_only=True)  # pyright: ignore[reportCallIssue]
        self._tag_names = self._load_tags(tags_path)

        logger.info(
            "WD-Tagger loaded (device=%s, tags=%d, threshold=%.2f)",
            self._device,
            len(self._tag_names),
            self._threshold,
        )

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._model is not None

    @staticmethod
    def _load_tags(tags_path: str) -> list[str]:
        """Load tag names from the selected_tags.csv file.

        Parameters
        ----------
        tags_path : str
            Path to the CSV file.

        Returns
        -------
        list[str]
            Ordered list of tag names matching model output indices.
        """
        tag_names: list[str] = []
        with open(tags_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag_names.append(row["name"])
        return tag_names

    def predict(self, image: np.ndarray) -> dict[str, float]:
        """Predict tags for an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        dict[str, float]
            Tag name to confidence mapping, filtered by threshold.
        """
        if self._model is None or self._transform is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        # Convert ndarray to PIL for timm transforms
        pil_image = Image.fromarray(image)
        input_tensor: torch.Tensor = self._transform(pil_image).unsqueeze(0)  # type: ignore[union-attr]
        input_tensor = input_tensor.to(self._device)

        with torch.no_grad():
            output = self._model(input_tensor)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(output).cpu().numpy()[0]

        # Build filtered predictions
        predictions: dict[str, float] = {}
        for i, prob in enumerate(probs):
            if i < len(self._tag_names) and float(prob) >= self._threshold:
                predictions[self._tag_names[i]] = float(prob)

        return predictions

    @staticmethod
    def download() -> None:
        """Pre-download model files without loading."""
        import timm

        # Trigger timm model download
        timm.create_model(TIMM_MODEL_NAME, pretrained=True)
        # Download tags CSV
        hf_hub_download(REPO_ID, TAGS_FILENAME)
        logger.info("WD-Tagger model downloaded")
