# Copyright 2026 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Configuration system — TOML-based layered config with scoring presets."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loupe.analyzers.base import AnalyzerConfig

# -- Scoring presets --

SCORING_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {
        "composition": 1.0,
        "color": 1.0,
        "detail": 1.0,
        "lighting": 1.0,
        "subject": 1.0,
        "style": 0.5,
    },
    "composition": {
        "composition": 3.0,
        "color": 1.0,
        "detail": 1.0,
        "lighting": 1.0,
        "subject": 1.0,
        "style": 0.5,
    },
    "visual": {
        "composition": 1.0,
        "color": 2.0,
        "detail": 2.0,
        "lighting": 1.0,
        "subject": 1.0,
        "style": 0.5,
    },
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.toml"
USER_CONFIG_PATH = Path.home() / ".config" / "loupe" / "config.toml"


# -- Config models --


class ScoringConfig(BaseModel):
    """Scoring configuration — weights and preset selection."""

    preset: str = "balanced"
    weights: dict[str, float] = Field(default_factory=dict)

    def resolved_weights(self) -> dict[str, float]:
        """Return effective weights: explicit weights override preset defaults."""
        base = SCORING_PRESETS.get(self.preset, SCORING_PRESETS["balanced"]).copy()
        base.update(self.weights)
        return base


class AnalyzersConfig(BaseModel):
    """Container for per-analyzer configurations."""

    composition: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    color: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    detail: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    lighting: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    subject: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    style: AnalyzerConfig = Field(default_factory=AnalyzerConfig)

    def get(self, name: str) -> AnalyzerConfig:
        """Get config for an analyzer by dimension name."""
        return getattr(self, name, AnalyzerConfig())  # type: ignore[no-any-return]


class LoupeConfig(BaseModel):
    """Root configuration for the Loupe application."""

    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    analyzers: AnalyzersConfig = Field(default_factory=AnalyzersConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        base_val = merged.get(key)
        if isinstance(base_val, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(
                base_val,  # pyright: ignore[reportUnknownArgumentType]
                value,  # pyright: ignore[reportUnknownArgumentType]
            )
        else:
            merged[key] = value
    return merged


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file and return its contents as a dict."""
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(
    config_path: Path | None = None,
    preset: str | None = None,
) -> LoupeConfig:
    """Load configuration with layered overrides.

    Parameters
    ----------
    config_path : Path | None
        Optional user config file. Falls back to ``~/.config/loupe/config.toml``
        if it exists.
    preset : str | None
        Optional scoring preset override (applied after file-based config).

    Returns
    -------
    LoupeConfig
        Fully resolved configuration.
    """
    # Layer 1: defaults
    data: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        data = _load_toml(DEFAULT_CONFIG_PATH)

    # Layer 2: user config
    user_path = config_path or USER_CONFIG_PATH
    if user_path.exists():
        user_data = _load_toml(user_path)
        data = _deep_merge(data, user_data)

    config = LoupeConfig.model_validate(data)

    # Layer 3: CLI preset override
    if preset is not None:
        config.scoring.preset = preset
        config.scoring.weights = {}

    return config
