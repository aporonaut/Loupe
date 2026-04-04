"""Tests for the anime detection model wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from loupe.models.detection import AnimeDetector, _nms


class TestNMS:
    """Tests for Non-Maximum Suppression."""

    def test_empty_input(self) -> None:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        assert _nms(boxes, scores, 0.5) == []

    def test_single_box(self) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        scores = np.array([0.9])
        assert _nms(boxes, scores, 0.5) == [0]

    def test_non_overlapping_boxes(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 100.0, 110.0, 110.0],
            ]
        )
        scores = np.array([0.9, 0.8])
        result = _nms(boxes, scores, 0.5)
        assert len(result) == 2

    def test_overlapping_boxes_suppressed(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 100.0, 100.0],
                [5.0, 5.0, 105.0, 105.0],  # Heavily overlaps with first
            ]
        )
        scores = np.array([0.9, 0.8])
        result = _nms(boxes, scores, 0.5)
        assert len(result) == 1
        assert result[0] == 0  # Higher score kept

    def test_partial_overlap_below_threshold(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 50.0, 50.0],
                [40.0, 40.0, 90.0, 90.0],  # Partial overlap
            ]
        )
        scores = np.array([0.9, 0.8])
        # IoU is small enough to keep both with high threshold
        result = _nms(boxes, scores, 0.9)
        assert len(result) == 2


class TestAnimeDetectorPreprocess:
    """Tests for detection preprocessing."""

    def test_not_loaded_raises(self) -> None:
        detector = AnimeDetector(gpu=False)
        with pytest.raises(RuntimeError, match="No models loaded"):
            detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_is_loaded_property(self) -> None:
        detector = AnimeDetector(gpu=False)
        assert not detector.is_loaded

    def test_preprocess_shape_aligned(self) -> None:
        detector = AnimeDetector(gpu=False)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, _scale_x, _scale_y = detector._preprocess(image)

        # Shape should be (1, 3, H, W) with H, W multiples of 32
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3
        assert tensor.shape[2] % 32 == 0
        assert tensor.shape[3] % 32 == 0

    def test_preprocess_normalization(self) -> None:
        detector = AnimeDetector(gpu=False)
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        tensor, _, _ = detector._preprocess(image)

        # Values should be in [0, 1]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_large_image_downscaled(self) -> None:
        detector = AnimeDetector(gpu=False)
        image = np.zeros((2000, 3000, 3), dtype=np.uint8)
        tensor, _, _ = detector._preprocess(image)

        # Should be downscaled to fit within MAX_INFER_SIZE
        assert tensor.shape[2] <= 1216 + 32  # Allow alignment overshoot
        assert tensor.shape[3] <= 1216 + 32

    def test_preprocess_scale_factors(self) -> None:
        detector = AnimeDetector(gpu=False)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, scale_x, scale_y = detector._preprocess(image)

        # Scale factors should map model coords back to original
        assert scale_x == 200 / tensor.shape[3]
        assert scale_y == 100 / tensor.shape[2]


class TestAnimeDetectorDecodeOutput:
    """Tests for YOLO output decoding."""

    def test_decode_no_detections_above_threshold(self) -> None:
        detector = AnimeDetector(gpu=False)
        # Output: (1, 5, 3) — 3 candidates all below threshold
        output = np.array(
            [[[10, 20, 30], [10, 20, 30], [5, 5, 5], [5, 5, 5], [0.1, 0.1, 0.1]]]
        )
        result = detector._decode_yolo_output(output, 1.0, 1.0, 0.5, "face")
        assert result == []

    def test_decode_single_detection(self) -> None:
        detector = AnimeDetector(gpu=False)
        # One detection: cx=50, cy=50, w=20, h=20, score=0.9
        output = np.array([[[50.0], [50.0], [20.0], [20.0], [0.9]]])
        result = detector._decode_yolo_output(output, 1.0, 1.0, 0.5, "face")

        assert len(result) == 1
        box = result[0]
        assert box.label == "face"
        assert box.confidence == pytest.approx(0.9)
        assert box.x1 == pytest.approx(40.0)
        assert box.y1 == pytest.approx(40.0)
        assert box.x2 == pytest.approx(60.0)
        assert box.y2 == pytest.approx(60.0)

    def test_decode_applies_scale(self) -> None:
        detector = AnimeDetector(gpu=False)
        output = np.array([[[50.0], [50.0], [20.0], [20.0], [0.9]]])
        result = detector._decode_yolo_output(output, 2.0, 3.0, 0.5, "person")

        assert len(result) == 1
        box = result[0]
        # x coords scaled by 2, y coords scaled by 3
        # Box: cx=50, cy=50, w=20, h=20 → x1=40, y1=40, x2=60, y2=60
        # Scaled: x1=80, y1=120, x2=120, y2=180
        assert box.x1 == pytest.approx(80.0)
        assert box.y1 == pytest.approx(120.0)
        assert box.x2 == pytest.approx(120.0)
        assert box.y2 == pytest.approx(180.0)

    def test_decode_nms_suppresses_duplicates(self) -> None:
        detector = AnimeDetector(gpu=False)
        # Two nearly identical detections — NMS should suppress one
        output = np.array(
            [
                [
                    [50.0, 51.0],  # cx
                    [50.0, 51.0],  # cy
                    [40.0, 40.0],  # w
                    [40.0, 40.0],  # h
                    [0.9, 0.85],  # score
                ]
            ]
        )
        result = detector._decode_yolo_output(output, 1.0, 1.0, 0.5, "head")

        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)


class TestAnimeDetectorPredict:
    """Tests for full prediction pipeline with mocked sessions."""

    @patch("loupe.models.detection.create_onnx_session")
    @patch("loupe.models.detection.download_json")
    @patch("loupe.models.detection.download_model")
    def test_predict_combines_models(
        self,
        mock_download: MagicMock,
        mock_json: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        mock_json.return_value = {"threshold": 0.3}

        # Create mock sessions for face and head
        def make_session(label: str) -> MagicMock:
            session = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "images"
            mock_output = MagicMock()
            mock_output.name = "output0"
            session.get_inputs.return_value = [mock_input]
            session.get_outputs.return_value = [mock_output]
            # Each model finds one detection: cx=50, cy=50, w=20, h=20, score=0.8
            session.run.return_value = [
                np.array([[[50.0], [50.0], [20.0], [20.0], [0.8]]])
            ]
            return session

        mock_create.side_effect = [make_session("face"), make_session("head")]

        detector = AnimeDetector(models=["face", "head"], gpu=False)
        detector.load()

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.predict(image)

        # Should have one detection from each model
        assert len(detections) == 2
        labels = {d.label for d in detections}
        assert "face" in labels
        assert "head" in labels

    @patch("loupe.models.detection.download_model")
    def test_download_static(self, mock_download: MagicMock) -> None:
        AnimeDetector.download(models=["face"])
        assert mock_download.call_count >= 1
