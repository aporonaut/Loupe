"""Tests for the ModelManager — model lifecycle and shared inference."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from loupe.analyzers.base import AnalyzerConfig
from loupe.core.config import AnalyzersConfig, LoupeConfig
from loupe.models.manager import ANALYZER_MODEL_DEPS, ModelManager


def _config_with_enabled(**enabled: bool) -> LoupeConfig:
    """Build a LoupeConfig with specific analyzers enabled/disabled."""
    analyzer_kwargs: dict[str, AnalyzerConfig] = {}
    for name in ("composition", "color", "detail", "lighting", "subject", "style"):
        analyzer_kwargs[name] = AnalyzerConfig(enabled=enabled.get(name, False))
    return LoupeConfig(analyzers=AnalyzersConfig(**analyzer_kwargs))


class TestAnalyzerModelDeps:
    """Tests for the model dependency mapping."""

    def test_classical_analyzers_have_no_deps(self) -> None:
        assert ANALYZER_MODEL_DEPS["composition"] == set()
        assert ANALYZER_MODEL_DEPS["color"] == set()

    def test_detail_needs_segmentation(self) -> None:
        assert "segmentation" in ANALYZER_MODEL_DEPS["detail"]

    def test_subject_needs_segmentation_and_detection(self) -> None:
        deps = ANALYZER_MODEL_DEPS["subject"]
        assert "segmentation" in deps
        assert "detection" in deps

    def test_style_needs_tagger_clip_aesthetic_segmentation(self) -> None:
        deps = ANALYZER_MODEL_DEPS["style"]
        assert "tagger" in deps
        assert "clip" in deps
        assert "aesthetic" in deps
        assert "segmentation" in deps


class TestModelManagerDetermineModels:
    """Tests for model requirement determination."""

    def test_only_classical_no_models_needed(self) -> None:
        config = _config_with_enabled(composition=True, color=True)
        manager = ModelManager(config, gpu=False)
        required = manager._determine_required_models()
        assert required == set()

    def test_detail_needs_segmentation(self) -> None:
        config = _config_with_enabled(detail=True)
        manager = ModelManager(config, gpu=False)
        required = manager._determine_required_models()
        assert required == {"segmentation"}

    def test_subject_needs_segmentation_and_detection(self) -> None:
        config = _config_with_enabled(subject=True)
        manager = ModelManager(config, gpu=False)
        required = manager._determine_required_models()
        assert required == {"segmentation", "detection"}

    def test_style_needs_tagger_clip_aesthetic_segmentation(self) -> None:
        config = _config_with_enabled(style=True)
        manager = ModelManager(config, gpu=False)
        required = manager._determine_required_models()
        assert required == {"tagger", "clip", "aesthetic", "segmentation"}

    def test_all_analyzers_all_models(self) -> None:
        config = _config_with_enabled(
            composition=True,
            color=True,
            detail=True,
            lighting=True,
            subject=True,
            style=True,
        )
        manager = ModelManager(config, gpu=False)
        required = manager._determine_required_models()
        assert required == {"segmentation", "tagger", "detection", "clip", "aesthetic"}

    def test_no_analyzers_no_models(self) -> None:
        config = _config_with_enabled()
        manager = ModelManager(config, gpu=False)
        required = manager._determine_required_models()
        assert required == set()


class TestModelManagerLoad:
    """Tests for model loading with mocked model classes."""

    @patch("loupe.models.segmentation.AnimeSegmentation")
    def test_loads_segmentation_when_needed(self, mock_seg_cls: MagicMock) -> None:
        config = _config_with_enabled(detail=True)
        manager = ModelManager(config, gpu=False)

        mock_seg_instance = MagicMock()
        mock_seg_cls.return_value = mock_seg_instance

        manager.load()

        mock_seg_cls.assert_called_once_with(gpu=False)
        mock_seg_instance.load.assert_called_once()

    @patch("loupe.models.clip.CLIPModel")
    @patch("loupe.models.aesthetic.AnimeAestheticScorer")
    @patch("loupe.models.tagger.WDTagger")
    @patch("loupe.models.segmentation.AnimeSegmentation")
    def test_loads_style_models(
        self,
        mock_seg_cls: MagicMock,
        mock_tagger_cls: MagicMock,
        mock_aesthetic_cls: MagicMock,
        mock_clip_cls: MagicMock,
    ) -> None:
        config = _config_with_enabled(style=True)
        manager = ModelManager(config, gpu=False)

        for cls in (mock_seg_cls, mock_tagger_cls, mock_aesthetic_cls, mock_clip_cls):
            cls.return_value = MagicMock()

        manager.load()

        mock_seg_cls.assert_called_once()
        mock_tagger_cls.assert_called_once()
        mock_aesthetic_cls.assert_called_once()
        mock_clip_cls.assert_called_once()

    def test_no_load_when_only_classical(self) -> None:
        config = _config_with_enabled(composition=True, color=True)
        manager = ModelManager(config, gpu=False)
        # Should complete without error and without importing model modules
        manager.load()


class TestModelManagerInference:
    """Tests for shared model inference."""

    def test_empty_shared_models_when_no_models(self) -> None:
        config = _config_with_enabled(composition=True, color=True)
        manager = ModelManager(config, gpu=False)
        manager.load()

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        shared = manager.run_shared_inference(image)

        assert "segmentation_mask" not in shared
        assert "tagger_predictions" not in shared
        assert "detection_boxes" not in shared
        assert "clip_embedding" not in shared

    @patch("loupe.models.segmentation.AnimeSegmentation")
    def test_segmentation_in_shared_output(self, mock_seg_cls: MagicMock) -> None:
        config = _config_with_enabled(detail=True)
        manager = ModelManager(config, gpu=False)

        mock_seg = MagicMock()
        mock_seg.predict.return_value = np.ones((100, 100), dtype=np.float32)
        mock_seg_cls.return_value = mock_seg

        manager.load()

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        shared = manager.run_shared_inference(image)

        assert "segmentation_mask" in shared
        np.testing.assert_array_equal(
            shared["segmentation_mask"], np.ones((100, 100), dtype=np.float32)
        )
        mock_seg.predict.assert_called_once()

    def test_aesthetic_scorer_accessible(self) -> None:
        config = _config_with_enabled(composition=True)
        manager = ModelManager(config, gpu=False)
        assert manager.aesthetic_scorer is None


class TestModelManagerDownloadAll:
    """Tests for the download_all static method."""

    @patch("loupe.models.clip.CLIPModel")
    @patch("loupe.models.aesthetic.AnimeAestheticScorer")
    @patch("loupe.models.detection.AnimeDetector")
    @patch("loupe.models.tagger.WDTagger")
    @patch("loupe.models.segmentation.AnimeSegmentation")
    def test_download_all_calls_all_models(
        self,
        mock_seg: MagicMock,
        mock_tagger: MagicMock,
        mock_detector: MagicMock,
        mock_aesthetic: MagicMock,
        mock_clip: MagicMock,
    ) -> None:
        ModelManager.download_all()

        mock_seg.download.assert_called_once()
        mock_tagger.download.assert_called_once()
        mock_detector.download.assert_called_once()
        mock_aesthetic.download.assert_called_once()
        mock_clip.download.assert_called_once()
