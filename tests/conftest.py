"""Shared test fixtures — synthetic images and temporary directories."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from loupe.core.models import (
    AnalyzerResult,
    ImageMetadata,
    LoupeResult,
    ScoringMetadata,
    Tag,
)


@pytest.fixture
def solid_red_image(tmp_path: Path) -> Path:
    """Create a solid red 100x100 PNG image."""
    img = Image.fromarray(
        np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8), mode="RGB"
    )
    path = tmp_path / "red.png"
    img.save(path)
    return path


@pytest.fixture
def gradient_image(tmp_path: Path) -> Path:
    """Create a horizontal gradient 200x100 PNG image."""
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 200, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = tmp_path / "gradient.png"
    img.save(path)
    return path


@pytest.fixture
def jpeg_image(tmp_path: Path) -> Path:
    """Create a simple JPEG image."""
    arr = np.random.default_rng(42).integers(0, 255, (80, 120, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = tmp_path / "test.jpg"
    img.save(path)
    return path


@pytest.fixture
def sample_loupe_result(tmp_path: Path) -> LoupeResult:
    """Create a sample LoupeResult for testing."""
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(image_path)
    return LoupeResult(
        image_path=image_path,
        image_metadata=ImageMetadata(width=50, height=50, format="PNG"),
        analyzer_results=[
            AnalyzerResult(
                analyzer="composition",
                score=0.8,
                tags=[
                    Tag(
                        name="rule_of_thirds",
                        confidence=0.9,
                        category="composition",
                    )
                ],
                metadata={"thirds_score": 0.9},
            ),
            AnalyzerResult(
                analyzer="color",
                score=0.6,
                tags=[
                    Tag(
                        name="harmonic_analogous",
                        confidence=0.7,
                        category="color",
                    )
                ],
            ),
        ],
        aggregate_score=0.7,
        scoring=ScoringMetadata(
            weights={"composition": 0.5, "color": 0.5},
            contributions={"composition": 0.4, "color": 0.3},
        ),
    )
