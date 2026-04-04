# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Pydantic data models — Tag, AnalyzerResult, ScoringMetadata, LoupeResult."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from loupe import __version__

# -- Supporting types --

AnalyzerCategory = Literal[
    "composition", "color", "detail", "lighting", "subject", "style"
]

SCHEMA_VERSION = "1.0"


# -- Models --


class ImageMetadata(BaseModel):
    """Metadata extracted from the source image."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str


class Tag(BaseModel):
    """A single aesthetic label produced by an analyzer."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: AnalyzerCategory


class AnalyzerResult(BaseModel):
    """Output of a single analyzer module."""

    analyzer: str
    score: float = Field(ge=0.0, le=1.0)
    tags: list[Tag] = []
    metadata: dict[str, Any] = {}


class ScoringMetadata(BaseModel):
    """Metadata about how the aggregate score was computed."""

    method: str = "weighted_mean"
    version: str = "1.0"
    weights: dict[str, float] = Field(default_factory=dict)
    contributions: dict[str, float] = Field(default_factory=dict)
    reliable: bool = True


class LoupeResult(BaseModel):
    """Complete analysis result for a single image."""

    image_path: Path
    image_metadata: ImageMetadata
    analyzer_results: list[AnalyzerResult] = []
    aggregate_score: float = Field(ge=0.0, le=1.0)
    scoring: ScoringMetadata = Field(default_factory=ScoringMetadata)
    timestamp: datetime = Field(default_factory=datetime.now)
    loupe_version: str = __version__
    schema_version: str = SCHEMA_VERSION
