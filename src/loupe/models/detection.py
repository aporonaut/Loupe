# Copyright 2025 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Anime detection models — face, head, and person detection via YOLOv8.

Uses deepghs ONNX models for detecting anime faces, heads, and persons.
All three models are YOLOv8s architecture with single-class output.

Output is a list of DetectionBox objects with normalized coordinates
relative to the original image dimensions.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import cv2
import numpy as np

from loupe.analyzers.base import DetectionBox
from loupe.models.onnx_utils import create_onnx_session, download_json, download_model

logger = logging.getLogger(__name__)

# Model configurations: repo_id, variant directory, label
DETECTION_MODELS: dict[str, tuple[str, str, str]] = {
    "face": (
        "deepghs/anime_face_detection",
        "face_detect_v1.4_s",
        "face",
    ),
    "head": (
        "deepghs/anime_head_detection",
        "head_detect_v2.0_s",
        "head",
    ),
    "person": (
        "deepghs/anime_person_detection",
        "person_detect_v1.3_s",
        "person",
    ),
}

# Default confidence thresholds (from model threshold.json files)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "face": 0.307,
    "head": 0.413,
    "person": 0.324,
}

NMS_IOU_THRESHOLD = 0.7
MAX_INFER_SIZE = 1216
ALIGN_TO = 32


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Greedy Non-Maximum Suppression.

    Parameters
    ----------
    boxes : np.ndarray
        (N, 4) array of [x1, y1, x2, y2] boxes.
    scores : np.ndarray
        (N,) confidence scores.
    iou_threshold : float
        IoU threshold for suppression.

    Returns
    -------
    list[int]
        Indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))

        if len(order) == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter)

        order = rest[iou <= iou_threshold]

    return keep


class AnimeDetector:
    """Anime face/head/person detector via YOLOv8 (ONNX).

    Loads one or more detection model variants and runs them
    sequentially, returning a combined list of detections.

    Parameters
    ----------
    models : list[str]
        Which detection models to load. Options: ``"face"``,
        ``"head"``, ``"person"``.
    gpu : bool
        Use CUDA if available.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        *,
        gpu: bool = True,
    ) -> None:
        self._model_keys = models or ["face", "head", "person"]
        self._gpu = gpu
        self._sessions: dict[str, Any] = {}
        self._thresholds: dict[str, float] = {}
        self._labels: dict[str, str] = {}

    def load(self) -> None:
        """Download and load all configured detection models."""
        for key in self._model_keys:
            if key not in DETECTION_MODELS:
                logger.warning("Unknown detection model: %s", key)
                continue

            repo_id, variant, label = DETECTION_MODELS[key]
            onnx_path = download_model(
                repo_id, f"{variant}/model.onnx", local_only=True
            )
            self._sessions[key] = create_onnx_session(onnx_path, gpu=self._gpu)
            self._labels[key] = label

            # Try to load threshold from model repo
            try:
                threshold_data = download_json(
                    repo_id, f"{variant}/threshold.json", local_only=True
                )
                self._thresholds[key] = float(
                    threshold_data.get("threshold", DEFAULT_THRESHOLDS.get(key, 0.3))
                )
            except Exception:
                self._thresholds[key] = DEFAULT_THRESHOLDS.get(key, 0.3)

            logger.info(
                "Detection model loaded: %s (threshold=%.3f)",
                key,
                self._thresholds[key],
            )

    @property
    def is_loaded(self) -> bool:
        """Whether at least one model has been loaded."""
        return len(self._sessions) > 0

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Preprocess image for YOLO inference.

        Resizes to fit within MAX_INFER_SIZE, aligning dimensions to
        multiples of ALIGN_TO.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        tuple[np.ndarray, float, float]
            (input_tensor, scale_x, scale_y) where scales map from
            model coordinates back to original image coordinates.
        """
        h0, w0 = image.shape[:2]

        # Scale to fit within max inference size
        scale = min(MAX_INFER_SIZE / max(h0, w0), 1.0)
        new_h = int(h0 * scale)
        new_w = int(w0 * scale)

        # Align to multiples of ALIGN_TO
        new_h = max(((new_h + ALIGN_TO - 1) // ALIGN_TO) * ALIGN_TO, ALIGN_TO)
        new_w = max(((new_w + ALIGN_TO - 1) // ALIGN_TO) * ALIGN_TO, ALIGN_TO)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0

        # CHW + batch
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, :]

        scale_x = w0 / new_w
        scale_y = h0 / new_h
        return tensor, scale_x, scale_y

    def _decode_yolo_output(
        self,
        output: np.ndarray,
        scale_x: float,
        scale_y: float,
        conf_threshold: float,
        label: str,
    ) -> list[DetectionBox]:
        """Decode YOLOv8 single-class output into DetectionBox objects.

        Parameters
        ----------
        output : np.ndarray
            Raw model output, shape (1, 5, N) for single-class.
        scale_x : float
            X scale factor to original image.
        scale_y : float
            Y scale factor to original image.
        conf_threshold : float
            Minimum confidence for detections.
        label : str
            Detection label (e.g. "face").

        Returns
        -------
        list[DetectionBox]
            Filtered and NMS-processed detections.
        """
        # Remove batch dim: (5, N)
        out = output[0]

        # Scores are at index 4 for single-class
        scores = out[4, :]

        # Filter by confidence
        mask = scores >= conf_threshold
        if not np.any(mask):
            return []

        filtered = out[:, mask]
        filtered_scores = filtered[4, :]

        # Convert cx, cy, w, h to x1, y1, x2, y2
        cx = filtered[0, :]
        cy = filtered[1, :]
        w = filtered[2, :]
        h = filtered[3, :]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        keep = _nms(boxes, filtered_scores, NMS_IOU_THRESHOLD)

        # Build DetectionBox list with coordinates scaled to original image
        detections: list[DetectionBox] = []
        for idx in keep:
            detections.append(
                DetectionBox(
                    label=label,
                    x1=float(boxes[idx, 0] * scale_x),
                    y1=float(boxes[idx, 1] * scale_y),
                    x2=float(boxes[idx, 2] * scale_x),
                    y2=float(boxes[idx, 3] * scale_y),
                    confidence=float(filtered_scores[idx]),
                )
            )

        return detections

    def predict(self, image: np.ndarray) -> list[DetectionBox]:
        """Run all loaded detection models on an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        list[DetectionBox]
            All detections across all loaded models.
        """
        if not self._sessions:
            msg = "No models loaded. Call load() first."
            raise RuntimeError(msg)

        tensor, scale_x, scale_y = self._preprocess(image)

        all_detections: list[DetectionBox] = []
        for key, session in self._sessions.items():
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            output = session.run([output_name], {input_name: tensor})

            detections = self._decode_yolo_output(
                output[0],
                scale_x,
                scale_y,
                self._thresholds[key],
                self._labels[key],
            )
            all_detections.extend(detections)

        return all_detections

    @staticmethod
    def download(models: list[str] | None = None) -> None:
        """Pre-download model files without loading.

        Parameters
        ----------
        models : list[str] | None
            Which models to download. Defaults to all three.
        """
        keys = models or list(DETECTION_MODELS.keys())
        for key in keys:
            if key not in DETECTION_MODELS:
                continue
            repo_id, variant, _ = DETECTION_MODELS[key]
            download_model(repo_id, f"{variant}/model.onnx")
            with contextlib.suppress(Exception):
                download_model(repo_id, f"{variant}/threshold.json")
        logger.info("Detection models downloaded: %s", keys)
