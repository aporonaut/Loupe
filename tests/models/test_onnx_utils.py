"""Tests for ONNX model loading utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from loupe.models.onnx_utils import (
    create_onnx_session,
    download_json,
    download_model,
    onnx_inference,
)


class TestDownloadModel:
    """Tests for HuggingFace Hub model downloading."""

    @patch("loupe.models.onnx_utils.hf_hub_download")
    def test_download_returns_path(self, mock_download: MagicMock) -> None:
        mock_download.return_value = "/cache/models/model.onnx"
        result = download_model("test/repo", "model.onnx")
        assert result == Path("/cache/models/model.onnx")
        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.onnx",
            revision=None,
            local_files_only=False,
        )

    @patch("loupe.models.onnx_utils.hf_hub_download")
    def test_download_with_revision(self, mock_download: MagicMock) -> None:
        mock_download.return_value = "/cache/models/model.onnx"
        download_model("test/repo", "model.onnx", revision="v1.0")
        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.onnx",
            revision="v1.0",
            local_files_only=False,
        )


class TestDownloadJson:
    """Tests for JSON file downloading and parsing."""

    @patch("loupe.models.onnx_utils.hf_hub_download")
    def test_download_json_parses(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        json_path = tmp_path / "meta.json"
        json_path.write_text('{"labels": ["a", "b"], "threshold": 0.5}')
        mock_download.return_value = str(json_path)

        result = download_json("test/repo", "meta.json")
        assert result == {"labels": ["a", "b"], "threshold": 0.5}


class TestCreateOnnxSession:
    """Tests for ONNX session creation.

    These tests mock the onnxruntime module at the import level since
    onnxruntime may not be installed in the dev environment (it's an
    optional dependency in the ``models`` extra).
    """

    def test_gpu_preferred(self, tmp_path: Path) -> None:
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            create_onnx_session(model_path, gpu=True)

        mock_ort.InferenceSession.assert_called_once_with(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_cpu_fallback_when_no_cuda(self, tmp_path: Path) -> None:
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            create_onnx_session(model_path, gpu=True)

        mock_ort.InferenceSession.assert_called_once_with(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def test_cpu_only(self, tmp_path: Path) -> None:
        mock_ort = MagicMock()
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            create_onnx_session(model_path, gpu=False)

        mock_ort.InferenceSession.assert_called_once_with(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )


class TestOnnxInference:
    """Tests for single-input/output ONNX inference."""

    def test_inference_with_auto_names(self) -> None:
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.array([1.0, 2.0])]

        result = onnx_inference(
            mock_session, np.zeros((1, 3, 64, 64), dtype=np.float32)
        )
        np.testing.assert_array_equal(result, [1.0, 2.0])
        mock_session.run.assert_called_once()

    def test_inference_with_explicit_names(self) -> None:
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([3.0])]

        result = onnx_inference(
            mock_session,
            np.zeros((1, 3, 64, 64), dtype=np.float32),
            input_name="img",
            output_name="mask",
        )
        mock_session.run.assert_called_once_with(
            ["mask"], {"img": pytest.approx(np.zeros((1, 3, 64, 64), dtype=np.float32))}
        )
        np.testing.assert_array_equal(result, [3.0])
