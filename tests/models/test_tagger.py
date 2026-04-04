"""Tests for the WD-Tagger v3 model wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
import torch

from loupe.models.tagger import WDTagger


class TestWDTaggerTagLoading:
    """Tests for tag vocabulary loading."""

    def test_load_tags_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "selected_tags.csv"
        csv_path.write_text(
            "tag_id,name,category,count\n0,1girl,0,5000\n1,blue_hair,0,3000\n"
        )
        tags = WDTagger._load_tags(str(csv_path))
        assert tags == ["1girl", "blue_hair"]

    def test_load_tags_empty_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "selected_tags.csv"
        csv_path.write_text("tag_id,name,category,count\n")
        tags = WDTagger._load_tags(str(csv_path))
        assert tags == []


def _setup_tagger(
    tag_names: list[str],
    logits: list[float],
    threshold: float = 0.35,
) -> WDTagger:
    """Create a WDTagger with mocked internals (skipping load)."""
    tagger = WDTagger(gpu=False, threshold=threshold)

    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([logits])
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    mock_transform = MagicMock()
    mock_transform.return_value = torch.zeros(3, 448, 448)

    tagger._model = mock_model
    tagger._transform = mock_transform
    tagger._tag_names = tag_names
    tagger._device = torch.device("cpu")

    return tagger


class TestWDTaggerPredict:
    """Tests for WD-Tagger prediction with mocked model."""

    def test_not_loaded_raises(self) -> None:
        tagger = WDTagger(gpu=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            tagger.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_is_loaded_property(self) -> None:
        tagger = WDTagger(gpu=False)
        assert not tagger.is_loaded

    def test_predict_filters_by_threshold(self) -> None:
        # sigmoid(2.2) ≈ 0.9, sigmoid(-2.2) ≈ 0.1, sigmoid(0.4) ≈ 0.6
        tagger = _setup_tagger(
            tag_names=["1girl", "blue_hair", "smile"],
            logits=[2.2, -2.2, 0.4],
            threshold=0.35,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = tagger.predict(image)

        # "1girl" (0.9) and "smile" (0.6) should pass threshold of 0.35
        # "blue_hair" (0.1) should be filtered out
        assert "1girl" in predictions
        assert "smile" in predictions
        assert "blue_hair" not in predictions
        assert predictions["1girl"] > 0.8
        assert predictions["smile"] > 0.5

    def test_predict_empty_above_threshold(self) -> None:
        """All predictions below threshold returns empty dict."""
        # sigmoid(-5) ≈ 0.007 — well below any threshold
        tagger = _setup_tagger(
            tag_names=["tag_a"],
            logits=[-5.0],
            threshold=0.5,
        )

        predictions = tagger.predict(np.zeros((50, 50, 3), dtype=np.uint8))
        assert predictions == {}

    def test_is_loaded_after_setup(self) -> None:
        tagger = _setup_tagger(["tag"], [0.0])
        assert tagger.is_loaded
