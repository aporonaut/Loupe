# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Analyzer protocol — interface all analyzers implement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    import numpy as np

    from loupe.core.models import AnalyzerResult


class DetectionBox(BaseModel):
    """Bounding box from a detection model."""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0.0, le=1.0)


class SharedModels(TypedDict, total=False):
    """Outputs from shared model inference, populated by the engine."""

    segmentation_mask: np.ndarray
    """Character mask from anime-segmentation (H, W) float32."""

    tagger_predictions: dict[str, float]
    """Tag name -> confidence from WD-Tagger."""

    detection_boxes: list[DetectionBox]
    """Face/head/person bounding boxes from detection models."""

    clip_embedding: np.ndarray
    """CLIP image embedding vector."""

    aesthetic_prediction: tuple[float, str, dict[str, float]]
    """(score, tier, tier_probabilities) from anime aesthetic scorer."""

    clip_style_scores: dict[str, float]
    """CLIP zero-shot style category probabilities."""


class AnalyzerConfig(BaseModel):
    """Configuration for a single analyzer."""

    enabled: bool = True
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class BaseAnalyzer(Protocol):
    """Protocol that all analyzer modules must satisfy."""

    name: str
    """Dimension name, e.g. 'composition'."""

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        """Analyze an image and return structured results.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array with shape (H, W, 3).
        config : AnalyzerConfig
            Per-analyzer configuration.
        shared : SharedModels
            Outputs from shared model inference.

        Returns
        -------
        AnalyzerResult
            Score, tags, and metadata for this dimension.
        """
        ...
