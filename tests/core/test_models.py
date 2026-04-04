"""Tests for core data models."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from loupe.core.models import (
    AnalyzerResult,
    ImageMetadata,
    LoupeResult,
    ScoringMetadata,
    Tag,
)


class TestTag:
    def test_valid_tag(self) -> None:
        tag = Tag(name="rule_of_thirds", confidence=0.85, category="composition")
        assert tag.name == "rule_of_thirds"
        assert tag.confidence == 0.85
        assert tag.category == "composition"

    def test_reject_out_of_range_confidence(self) -> None:
        with pytest.raises(ValidationError):
            Tag(name="test", confidence=1.5, category="color")
        with pytest.raises(ValidationError):
            Tag(name="test", confidence=-0.1, category="color")

    def test_reject_invalid_category(self) -> None:
        with pytest.raises(ValidationError):
            Tag(name="test", confidence=0.5, category="invalid")  # type: ignore[arg-type]

    def test_all_valid_categories(self) -> None:
        for cat in ("composition", "color", "detail", "lighting", "subject", "style"):
            tag = Tag(name="test", confidence=0.5, category=cat)  # type: ignore[arg-type]
            assert tag.category == cat


class TestAnalyzerResult:
    def test_valid_result(self) -> None:
        result = AnalyzerResult(
            analyzer="composition",
            score=0.72,
            tags=[Tag(name="balanced", confidence=0.8, category="composition")],
            metadata={"sub_scores": {"thirds": 0.8, "balance": 0.6}},
        )
        assert result.analyzer == "composition"
        assert result.score == 0.72
        assert len(result.tags) == 1

    def test_reject_out_of_range_score(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzerResult(analyzer="test", score=1.1)
        with pytest.raises(ValidationError):
            AnalyzerResult(analyzer="test", score=-0.1)

    def test_defaults(self) -> None:
        result = AnalyzerResult(analyzer="test", score=0.5)
        assert result.tags == []
        assert result.metadata == {}


class TestImageMetadata:
    def test_valid(self) -> None:
        meta = ImageMetadata(width=1920, height=1080, format="PNG")
        assert meta.width == 1920

    def test_reject_zero_dimensions(self) -> None:
        with pytest.raises(ValidationError):
            ImageMetadata(width=0, height=1080, format="PNG")


class TestScoringMetadata:
    def test_defaults(self) -> None:
        scoring = ScoringMetadata()
        assert scoring.method == "weighted_mean"
        assert scoring.version == "1.0"
        assert scoring.reliable is True

    def test_with_data(self) -> None:
        scoring = ScoringMetadata(
            weights={"composition": 0.5, "color": 0.5},
            contributions={"composition": 0.36, "color": 0.38},
        )
        assert scoring.weights["composition"] == 0.5


class TestLoupeResult:
    def test_construction(self) -> None:
        result = LoupeResult(
            image_path=Path("test.png"),
            image_metadata=ImageMetadata(width=1920, height=1080, format="PNG"),
            aggregate_score=0.0,
        )
        assert result.schema_version == "1.0"
        assert result.loupe_version == "0.1.0"

    def test_json_roundtrip(self) -> None:
        original = LoupeResult(
            image_path=Path("images/test.png"),
            image_metadata=ImageMetadata(width=1920, height=1080, format="PNG"),
            analyzer_results=[
                AnalyzerResult(
                    analyzer="composition",
                    score=0.72,
                    tags=[
                        Tag(
                            name="rule_of_thirds",
                            confidence=0.85,
                            category="composition",
                        )
                    ],
                    metadata={"thirds_score": 0.85},
                )
            ],
            aggregate_score=0.72,
            scoring=ScoringMetadata(
                weights={"composition": 1.0},
                contributions={"composition": 0.72},
            ),
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )
        json_str = original.model_dump_json()
        restored = LoupeResult.model_validate_json(json_str)

        assert restored.image_path == original.image_path
        assert restored.aggregate_score == original.aggregate_score
        assert restored.analyzer_results[0].score == 0.72
        assert restored.analyzer_results[0].tags[0].name == "rule_of_thirds"
        assert restored.timestamp == original.timestamp

    def test_json_is_valid_json(self) -> None:
        result = LoupeResult(
            image_path=Path("test.png"),
            image_metadata=ImageMetadata(width=100, height=100, format="PNG"),
            aggregate_score=0.5,
        )
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == "1.0"
