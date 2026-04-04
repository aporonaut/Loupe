# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Sidecar I/O — .loupe/ directory management, JSON read/write."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from loupe.core.models import LoupeResult

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SIDECAR_DIR = ".loupe"


def _sidecar_path(image_path: Path) -> Path:
    """Return the sidecar JSON path for a given image."""
    return image_path.parent / SIDECAR_DIR / f"{image_path.name}.json"


def has_result(image_path: Path) -> bool:
    """Check whether a sidecar result exists for the given image.

    Parameters
    ----------
    image_path : Path
        Path to the source image.

    Returns
    -------
    bool
        True if a sidecar file exists.
    """
    return _sidecar_path(image_path).exists()


def write_result(result: LoupeResult) -> Path:
    """Write a LoupeResult as a JSON sidecar file.

    Parameters
    ----------
    result : LoupeResult
        The analysis result to persist.

    Returns
    -------
    Path
        Path to the written sidecar file.
    """
    sidecar = _sidecar_path(result.image_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.debug("Wrote sidecar: %s", sidecar)
    return sidecar


def read_result(image_path: Path) -> LoupeResult | None:
    """Read a sidecar result for the given image.

    Parameters
    ----------
    image_path : Path
        Path to the source image.

    Returns
    -------
    LoupeResult | None
        The deserialized result, or None if no sidecar exists.
    """
    sidecar = _sidecar_path(image_path)
    if not sidecar.exists():
        return None
    try:
        return LoupeResult.model_validate_json(sidecar.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read sidecar: %s", sidecar, exc_info=True)
        return None
