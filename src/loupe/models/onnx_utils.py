# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Generic ONNX model loading utilities.

Provides thin wrappers around onnxruntime for session creation with
GPU-preferred / CPU-fallback execution, and HuggingFace Hub model
downloading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    local_only: bool = False,
) -> Path:
    """Download a model file from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (e.g. ``"deepghs/anime_aesthetic"``).
    filename : str
        File path within the repository (e.g. ``"model.onnx"``).
    revision : str | None
        Optional git revision (branch, tag, or commit hash).
    local_only : bool
        If True, only look in local cache — never make network requests.
        Raises an error if the file is not cached.

    Returns
    -------
    Path
        Local path to the downloaded file (in the HF cache).
    """
    local_path: str = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_files_only=local_only,
    )
    return Path(local_path)


def download_json(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    local_only: bool = False,
) -> dict[str, Any]:
    """Download and parse a JSON file from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID.
    filename : str
        JSON file path within the repository.
    revision : str | None
        Optional git revision.

    Returns
    -------
    dict[str, Any]
        Parsed JSON contents.
    """
    path = download_model(repo_id, filename, revision=revision, local_only=local_only)
    with path.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


_cuda_dlls_added = False


def _ensure_cuda_dlls() -> None:
    """Add PyTorch's lib directory to the DLL search path.

    ONNX Runtime needs cuDNN/CUDA DLLs that PyTorch bundles but
    doesn't expose on the system PATH. This adds PyTorch's lib
    directory so ONNX Runtime can find them (Windows only).
    """
    global _cuda_dlls_added
    if _cuda_dlls_added:
        return
    _cuda_dlls_added = True

    import sys

    if sys.platform != "win32":
        return

    try:
        import os

        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
            logger.debug("Added PyTorch DLL directory: %s", torch_lib)
    except Exception:
        logger.debug("Could not add PyTorch DLL directory", exc_info=True)


def create_onnx_session(
    model_path: Path,
    *,
    gpu: bool = True,
) -> Any:
    """Create an ONNX inference session with GPU preferred, CPU fallback.

    Parameters
    ----------
    model_path : Path
        Path to the ``.onnx`` model file.
    gpu : bool
        If True, attempt to use CUDAExecutionProvider first.

    Returns
    -------
    onnxruntime.InferenceSession
        Ready-to-use inference session.
    """
    _ensure_cuda_dlls()
    import onnxruntime as ort  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    providers: list[str] = []
    if gpu:
        available: list[str] = ort.get_available_providers()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            logger.info("ONNX using CUDAExecutionProvider")
        else:
            logger.info("CUDA not available for ONNX, falling back to CPU")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        str(model_path),
        providers=providers,
    )
    return session  # pyright: ignore[reportUnknownVariableType]


def onnx_inference(
    session: Any,
    input_array: np.ndarray,
    *,
    input_name: str | None = None,
    output_name: str | None = None,
) -> np.ndarray:
    """Run a single-input, single-output ONNX inference.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        The inference session.
    input_array : np.ndarray
        Input tensor (must match the model's expected shape and dtype).
    input_name : str | None
        ONNX input name. If None, uses the first input.
    output_name : str | None
        ONNX output name. If None, uses the first output.

    Returns
    -------
    np.ndarray
        Model output as a NumPy array.
    """
    if input_name is None:
        input_name = session.get_inputs()[0].name
    if output_name is None:
        output_name = session.get_outputs()[0].name

    results = session.run([output_name], {input_name: input_array})
    return results[0]  # type: ignore[no-any-return]
