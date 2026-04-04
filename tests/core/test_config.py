"""Tests for the configuration system."""

from pathlib import Path

import pytest

from loupe.core.config import (
    SCORING_PRESETS,
    LoupeConfig,
    ScoringConfig,
    _deep_merge,
    load_config,
)


class TestScoringConfig:
    def test_default_preset(self) -> None:
        config = ScoringConfig()
        assert config.preset == "balanced"

    def test_resolved_weights_balanced(self) -> None:
        config = ScoringConfig(preset="balanced")
        weights = config.resolved_weights()
        assert weights == SCORING_PRESETS["balanced"]

    def test_resolved_weights_composition(self) -> None:
        config = ScoringConfig(preset="composition")
        weights = config.resolved_weights()
        assert weights["composition"] == 3.0
        assert weights["color"] == 1.0

    def test_resolved_weights_visual(self) -> None:
        config = ScoringConfig(preset="visual")
        weights = config.resolved_weights()
        assert weights["color"] == 2.0
        assert weights["detail"] == 2.0

    def test_explicit_weight_overrides_preset(self) -> None:
        config = ScoringConfig(preset="balanced", weights={"composition": 5.0})
        weights = config.resolved_weights()
        assert weights["composition"] == 5.0
        assert weights["color"] == 1.0

    def test_unknown_preset_falls_back_to_balanced(self) -> None:
        config = ScoringConfig(preset="nonexistent")
        weights = config.resolved_weights()
        assert weights == SCORING_PRESETS["balanced"]


class TestDeepMerge:
    def test_simple(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        assert _deep_merge(base, override) == {"a": 1, "b": 3, "c": 4}

    def test_nested(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 2}})
        assert base["a"]["x"] == 1


class TestLoadConfig:
    def test_default_config_loads(self) -> None:
        config = load_config()
        assert isinstance(config, LoupeConfig)
        assert config.scoring.preset == "balanced"
        assert config.analyzers.composition.enabled is True

    def test_preset_override(self) -> None:
        config = load_config(preset="composition")
        assert config.scoring.preset == "composition"
        weights = config.scoring.resolved_weights()
        assert weights["composition"] == 3.0

    def test_user_config_override(self, tmp_path: Path) -> None:
        user_config = tmp_path / "config.toml"
        user_config.write_text(
            "[analyzers.color]\nenabled = false\nconfidence_threshold = 0.5\n"
        )
        config = load_config(config_path=user_config)
        assert config.analyzers.color.enabled is False
        assert config.analyzers.color.confidence_threshold == 0.5
        # Other analyzers unaffected
        assert config.analyzers.composition.enabled is True

    def test_nonexistent_user_config_ignored(self, tmp_path: Path) -> None:
        config = load_config(config_path=tmp_path / "nonexistent.toml")
        assert isinstance(config, LoupeConfig)

    def test_invalid_config_rejected(self, tmp_path: Path) -> None:
        bad_config = tmp_path / "bad.toml"
        bad_config.write_text("[analyzers.color]\nconfidence_threshold = 2.0\n")
        with pytest.raises(Exception):  # noqa: B017
            load_config(config_path=bad_config)

    def test_get_analyzer_config(self) -> None:
        config = load_config()
        assert config.analyzers.get("composition").enabled is True
