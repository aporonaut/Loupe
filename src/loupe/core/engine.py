# Copyright 2025 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Analysis engine — orchestrates analyzer dispatch and result aggregation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from loupe.core.models import AnalyzerResult, LoupeResult
from loupe.core.scoring import compute_aggregate
from loupe.io.image import load_image
from loupe.io.sidecar import has_result, write_result
from loupe.models.manager import ModelManager

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from loupe.analyzers.base import BaseAnalyzer, SharedModels
    from loupe.core.config import LoupeConfig

logger = logging.getLogger(__name__)


class Engine:
    """Analysis orchestrator — dispatches to analyzers and aggregates results.

    Parameters
    ----------
    config : LoupeConfig
        Application configuration.
    gpu : bool
        Use CUDA for model inference if available.
    """

    def __init__(self, config: LoupeConfig, *, gpu: bool = True) -> None:
        self._config = config
        self._analyzers: list[BaseAnalyzer] = []
        self._model_manager = ModelManager(config, gpu=gpu)
        self._models_loaded = False

    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """Register an analyzer for use during analysis.

        Parameters
        ----------
        analyzer : BaseAnalyzer
            An analyzer instance satisfying the protocol.
        """
        self._analyzers.append(analyzer)

    def ensure_models_loaded(self) -> None:
        """Load shared models if not already loaded.

        Called automatically on first analysis, but can be called
        explicitly to separate model loading from image processing
        (e.g. to load before starting a progress bar).
        """
        if not self._models_loaded:
            registered = {a.name for a in self._analyzers}
            self._model_manager.load(registered_analyzers=registered)
            self._models_loaded = True

    def _get_shared_models(self, image: np.ndarray) -> SharedModels:
        """Run shared model inference for a single image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array (H, W, 3).

        Returns
        -------
        SharedModels
            Shared model outputs for this image.
        """
        self.ensure_models_loaded()
        return self._model_manager.run_shared_inference(image)

    @property
    def model_manager(self) -> ModelManager:
        """Access the model manager (e.g. for aesthetic scorer)."""
        return self._model_manager

    def analyze(
        self,
        image_path: Path,
        *,
        force: bool = False,
    ) -> LoupeResult | None:
        """Analyze a single image.

        Parameters
        ----------
        image_path : Path
            Path to the image file.
        force : bool
            If True, re-analyze even if a sidecar already exists.

        Returns
        -------
        LoupeResult | None
            The analysis result, or None if the image could not be loaded.
        """
        image_path = image_path.resolve()

        if not force and has_result(image_path):
            logger.info("Skipping (already analyzed): %s", image_path.name)
            return None

        loaded = load_image(image_path)
        if loaded is None:
            return None

        shared = self._get_shared_models(loaded.array)

        results: list[AnalyzerResult] = []
        for analyzer in self._analyzers:
            config = self._config.analyzers.get(analyzer.name)
            if not config.enabled:
                continue
            result = analyzer.analyze(loaded.array, config, shared)
            results.append(result)

        weights = self._config.scoring.resolved_weights()
        aggregate, scoring = compute_aggregate(results, weights)

        loupe_result = LoupeResult(
            image_path=image_path,
            image_metadata=loaded.metadata,
            analyzer_results=results,
            aggregate_score=aggregate,
            scoring=scoring,
        )

        write_result(loupe_result)
        return loupe_result

    def analyze_batch(
        self,
        paths: list[Path],
        *,
        force: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[LoupeResult]:
        """Analyze a batch of images.

        Parameters
        ----------
        paths : list[Path]
            Image file paths.
        force : bool
            If True, re-analyze even if sidecars exist.
        progress_callback : Callable[[int, int], None] | None
            Optional callback called with (current_index, total_count).

        Returns
        -------
        list[LoupeResult]
            Results for successfully analyzed images.
        """
        results: list[LoupeResult] = []
        total = len(paths)
        for i, path in enumerate(paths):
            if progress_callback:
                progress_callback(i, total)
            result = self.analyze(path, force=force)
            if result is not None:
                results.append(result)
        if progress_callback:
            progress_callback(total, total)
        return results
