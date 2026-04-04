# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Image loading — format detection, metadata extraction, array conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from loupe.core.models import ImageMetadata

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "TIFF", "BMP"})


@dataclass(frozen=True)
class LoadedImage:
    """An image loaded and ready for analysis."""

    array: np.ndarray
    """RGB uint8 array with shape (H, W, 3)."""

    metadata: ImageMetadata


def load_image(path: Path) -> LoadedImage | None:
    """Load an image file and return its array and metadata.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    LoadedImage | None
        The loaded image, or None if the format is unsupported.
    """
    try:
        with Image.open(path) as img:
            fmt = img.format or ""
            if fmt.upper() not in SUPPORTED_FORMATS:
                logger.warning("Unsupported image format '%s': %s", fmt, path)
                return None

            metadata = ImageMetadata(
                width=img.width,
                height=img.height,
                format=fmt,
            )

            rgb = img.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)

    except Exception:
        logger.warning("Failed to load image: %s", path, exc_info=True)
        return None

    return LoadedImage(array=array, metadata=metadata)
