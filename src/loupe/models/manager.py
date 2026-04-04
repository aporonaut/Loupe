# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Model manager — loads and manages shared ML models for the analysis engine.

The ModelManager determines which models to load based on which analyzers
are enabled, runs shared model inference once per image, and provides
the results as a SharedModels dict for analyzers to consume.

Model-to-analyzer dependency mapping:
- segmentation_mask: detail, lighting, subject, style
- tagger_predictions: style, lighting (supplementary tags)
- detection_boxes: subject
- clip_embedding: style
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from loupe.analyzers.base import SharedModels

if TYPE_CHECKING:
    import numpy as np

    from loupe.core.config import LoupeConfig
    from loupe.models.aesthetic import AnimeAestheticScorer
    from loupe.models.clip import CLIPModel
    from loupe.models.detection import AnimeDetector
    from loupe.models.segmentation import AnimeSegmentation
    from loupe.models.tagger import WDTagger

logger = logging.getLogger(__name__)

# Which analyzers require which shared model outputs
ANALYZER_MODEL_DEPS: dict[str, set[str]] = {
    "composition": set(),
    "color": set(),
    "detail": {"segmentation"},
    "lighting": {"segmentation", "tagger"},
    "subject": {"segmentation", "detection"},
    "style": {"tagger", "clip", "aesthetic", "segmentation"},
}


class ModelManager:
    """Manages shared ML model lifecycle and inference.

    Loads only the models required by enabled analyzers. Provides
    ``run_shared_inference()`` to produce the ``SharedModels`` dict
    consumed by analyzers.

    Parameters
    ----------
    config : LoupeConfig
        Application configuration (used to determine which analyzers
        are enabled and thus which models to load).
    gpu : bool
        Use CUDA for model inference if available.
    """

    def __init__(self, config: LoupeConfig, *, gpu: bool = True) -> None:
        self._config = config
        self._gpu = gpu
        self._required_models: set[str] = set()

        self._segmentation: AnimeSegmentation | None = None
        self._tagger: WDTagger | None = None
        self._detector: AnimeDetector | None = None
        self._aesthetic: AnimeAestheticScorer | None = None
        self._clip: CLIPModel | None = None

    def _determine_required_models(
        self,
        registered_analyzers: set[str] | None = None,
    ) -> set[str]:
        """Determine which shared models are needed.

        Parameters
        ----------
        registered_analyzers : set[str] | None
            If provided, only consider these analyzer names. Otherwise,
            consider all enabled analyzers from config.
        """
        required: set[str] = set()
        for analyzer_name, deps in ANALYZER_MODEL_DEPS.items():
            # Only load models for analyzers that are both registered and enabled
            if (
                registered_analyzers is not None
                and analyzer_name not in registered_analyzers
            ):
                continue
            analyzer_config = self._config.analyzers.get(analyzer_name)
            if analyzer_config.enabled:
                required.update(deps)
        return required

    def load(self, registered_analyzers: set[str] | None = None) -> None:
        """Load all models required by registered and enabled analyzers.

        Models are loaded from local cache only — no network requests.
        Run ``loupe setup`` first to download models.

        Parameters
        ----------
        registered_analyzers : set[str] | None
            Names of analyzers registered in the engine. If provided,
            only loads models needed by these analyzers (intersected
            with what's enabled in config). If None, loads models for
            all enabled analyzers.
        """
        import os

        self._required_models = self._determine_required_models(registered_analyzers)

        if not self._required_models:
            logger.info("No shared models required by enabled analyzers")
            return

        logger.info("Loading shared models: %s", sorted(self._required_models))

        # Force offline mode so timm/open_clip use cached weights only
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._load_models()
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

    def _load_models(self) -> None:
        """Load required models (called with HF_HUB_OFFLINE=1)."""
        if "segmentation" in self._required_models:
            from loupe.models.segmentation import AnimeSegmentation

            self._segmentation = AnimeSegmentation(gpu=self._gpu)
            self._segmentation.load()

        if "tagger" in self._required_models:
            from loupe.models.tagger import WDTagger

            tagger_threshold = self._config.analyzers.get("style").params.get(
                "tagger_threshold", 0.35
            )
            self._tagger = WDTagger(
                gpu=self._gpu,
                threshold=float(tagger_threshold),
            )
            self._tagger.load()

        if "detection" in self._required_models:
            from loupe.models.detection import AnimeDetector

            self._detector = AnimeDetector(gpu=self._gpu)
            self._detector.load()

        if "aesthetic" in self._required_models:
            from loupe.models.aesthetic import AnimeAestheticScorer

            self._aesthetic = AnimeAestheticScorer(gpu=self._gpu)
            self._aesthetic.load()

        if "clip" in self._required_models:
            from loupe.models.clip import CLIPModel

            self._clip = CLIPModel(gpu=self._gpu)
            self._clip.load()

    def run_shared_inference(self, image: np.ndarray) -> SharedModels:
        """Run all loaded shared models on a single image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        SharedModels
            Dict of shared model outputs. Keys are only present for
            models that were loaded.
        """
        shared = SharedModels()

        if self._segmentation is not None:
            shared["segmentation_mask"] = self._segmentation.predict(image)

        if self._tagger is not None:
            shared["tagger_predictions"] = self._tagger.predict(image)

        if self._detector is not None:
            shared["detection_boxes"] = self._detector.predict(image)

        if self._clip is not None:
            image_emb = self._clip.get_image_embedding(image)
            shared["clip_embedding"] = image_emb
            # Zero-shot style classification reusing pre-computed embedding
            shared["clip_style_scores"] = self._clip.zero_shot_classify_from_embedding(
                image_emb,
                [
                    "naturalistic anime",
                    "geometric abstract anime",
                    "painterly anime",
                    "digital modern anime",
                    "retro cel anime",
                ],
            )

        if self._aesthetic is not None:
            shared["aesthetic_prediction"] = self._aesthetic.predict(image)

        return shared

    @property
    def aesthetic_scorer(self) -> AnimeAestheticScorer | None:
        """Direct access to the aesthetic scorer for the Style analyzer."""
        return self._aesthetic

    @staticmethod
    def download_all() -> None:
        """Pre-download all model files without loading them.

        Downloads every model regardless of configuration — intended
        for the ``loupe setup`` command.
        """
        from loupe.models.aesthetic import AnimeAestheticScorer
        from loupe.models.clip import CLIPModel
        from loupe.models.detection import AnimeDetector
        from loupe.models.segmentation import AnimeSegmentation
        from loupe.models.tagger import WDTagger

        logger.info("Downloading all models...")
        AnimeSegmentation.download()
        WDTagger.download()
        AnimeDetector.download()
        AnimeAestheticScorer.download()
        CLIPModel.download()
        logger.info("All models downloaded successfully")
