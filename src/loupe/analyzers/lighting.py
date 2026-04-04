# Copyright 2025 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Lighting analyzer — illumination design analysis.

Measures the quality and design of illumination in an anime frame across
six sub-properties: contrast ratio, light directionality, rim/edge
lighting, shadow quality, atmospheric lighting, and highlight/shadow
balance.

Uses the shared segmentation mask from SharedModels for rim-light
detection (boundary luminance differential). WD-Tagger predictions
supplement the tag output with learned lighting labels.

Tags produced:
    high_contrast — strong tonal range (contrast ratio > 0.7)
    low_contrast — narrow tonal range (contrast ratio < 0.3)
    dramatic_lighting — overall lighting score is very high (> 0.7)
    flat_lighting — overall lighting score is very low (< 0.25)
    rim_lit — rim/edge lighting detected on character boundary
    soft_shadows — shadow boundaries are gradual/soft
    hard_shadows — shadow boundaries are sharp/crisp
    atmospheric — bloom or glow effects detected
    high_key — predominantly bright tonality
    low_key — predominantly dark tonality
    balanced_exposure — even distribution across tonal zones
    directional_light — strong luminance gradient across frame

WD-Tagger supplementary tags (passed through when confident):
    backlighting, rim_lighting, sunlight, moonlight, spotlight,
    dramatic_lighting, lens_flare, light_rays, glowing

Known limitations:
    - Light directionality uses a coarse 3x3 grid, which can be
      ambiguous under multi-source or non-physical anime lighting.
    - Atmospheric bloom detection may false-positive on large bright
      regions that are not bloom effects (e.g. white backgrounds).
    - Rim-light detection requires a segmentation mask; without it,
      that sub-property defaults to 0.0.
    - Calibration is initial/theoretical — will be refined with real images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from loupe.core.models import AnalyzerResult, Tag

if TYPE_CHECKING:
    from loupe.analyzers.base import AnalyzerConfig, SharedModels

# -- Constants --

# Default sub-property weights from RQ2 §7.6
DEFAULT_WEIGHTS: dict[str, float] = {
    "contrast_ratio": 0.25,
    "light_directionality": 0.10,
    "rim_edge_lighting": 0.20,
    "shadow_quality": 0.15,
    "atmospheric_lighting": 0.15,
    "highlight_shadow_balance": 0.15,
}

# Segmentation mask binarization threshold
_MASK_THRESHOLD = 0.5

# Boundary dilation kernel size (pixels) for rim-light detection
_RIM_DILATION_PX = 7

# Minimum boundary pixels for meaningful rim-light measurement
_MIN_BOUNDARY_PIXELS = 50

# Shadow threshold on V channel (fraction of 255)
_SHADOW_V_THRESHOLD = 0.30

# Highlight threshold on V channel (fraction of 255)
_HIGHLIGHT_V_THRESHOLD = 0.70

# Atmospheric bloom: Gaussian blur kernel size
_BLOOM_KSIZE = 31

# Atmospheric bloom: intensity threshold (V difference to count as bloom)
_BLOOM_THRESHOLD = 20

# Directionality: minimum gradient ratio to consider "directional"
_DIRECTIONALITY_THRESHOLD = 1.5

# WD-Tagger lighting-relevant tag names
_TAGGER_LIGHTING_TAGS: set[str] = {
    "backlighting",
    "rim_lighting",
    "sunlight",
    "moonlight",
    "spotlight",
    "dramatic_lighting",
    "lens_flare",
    "light_rays",
    "glowing",
}


# -- Image Preparation --


def _extract_v_channel(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSV and return the V (value/brightness) channel.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).

    Returns
    -------
    np.ndarray
        V channel as uint8 (H, W).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 2]


def _get_binary_mask(shared: SharedModels, h: int, w: int) -> np.ndarray | None:
    """Extract binary segmentation mask, resized to (h, w).

    Parameters
    ----------
    shared : SharedModels
        Shared model outputs.
    h : int
        Target height.
    w : int
        Target width.

    Returns
    -------
    np.ndarray | None
        Binary uint8 mask (H, W) with 1=character, 0=background.
        None if no segmentation mask available.
    """
    seg_mask = shared.get("segmentation_mask")
    if seg_mask is None:
        return None

    if seg_mask.shape[:2] != (h, w):
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return (seg_mask > _MASK_THRESHOLD).astype(np.uint8)


# -- Sub-Property Measurements --


def measure_contrast_ratio(v_channel: np.ndarray) -> float:
    """Measure tonal range via robust percentile contrast.

    Computes ``percentile(V, 95) - percentile(V, 5)`` normalized to [0, 1].
    Robust to specular highlights and black outlines.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).

    Returns
    -------
    float
        Score in [0, 1]. Higher = wider tonal range.
    """
    p5 = float(np.percentile(v_channel, 5))
    p95 = float(np.percentile(v_channel, 95))
    contrast = (p95 - p5) / 255.0
    return float(np.clip(contrast, 0.0, 1.0))


def measure_light_directionality(v_channel: np.ndarray) -> float:
    """Measure strength of directional lighting via 3x3 grid luminance.

    Divides the V channel into a 3x3 grid, computes mean luminance per
    cell, and measures how asymmetric the luminance distribution is.
    Strong asymmetry indicates directional light.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).

    Returns
    -------
    float
        Score in [0, 1]. Higher = more directional lighting.
    """
    h, w = v_channel.shape
    grid_h = h // 3
    grid_w = w // 3

    if grid_h == 0 or grid_w == 0:
        return 0.0

    # Compute mean luminance for each cell in 3x3 grid
    cells: list[float] = []
    for row in range(3):
        for col in range(3):
            y0 = row * grid_h
            y1 = (row + 1) * grid_h if row < 2 else h
            x0 = col * grid_w
            x1 = (col + 1) * grid_w if col < 2 else w
            cells.append(float(np.mean(v_channel[y0:y1, x0:x1])))

    cells_arr = np.array(cells)
    cell_range = float(cells_arr.max() - cells_arr.min())

    # Normalize: a range of 80+ intensity units across cells is very directional
    # Typical: uniform ~5-15, moderate ~30-50, strong directional ~60+
    score = min(1.0, cell_range / 80.0)
    return float(np.clip(score, 0.0, 1.0))


def measure_rim_edge_lighting(v_channel: np.ndarray, mask: np.ndarray | None) -> float:
    """Detect rim/edge lighting via boundary luminance differential.

    Dilates the character mask to create a boundary ring, then compares
    mean luminance of the boundary ring to nearby background luminance.
    Positive differential indicates rim lighting.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).
    mask : np.ndarray | None
        Binary character mask (H, W) uint8. None if unavailable.

    Returns
    -------
    float
        Score in [0, 1]. Higher = stronger rim/edge lighting.
    """
    if mask is None or int(np.sum(mask)) == 0:
        return 0.0

    # Create boundary ring via dilation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_RIM_DILATION_PX, _RIM_DILATION_PX)
    )
    dilated: np.ndarray = cv2.dilate(mask, kernel, iterations=1)
    boundary = (dilated > 0) & (mask == 0)

    boundary_pixels = int(np.sum(boundary))
    if boundary_pixels < _MIN_BOUNDARY_PIXELS:
        return 0.0

    # Create a wider background ring (further from boundary)
    kernel_wide = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_RIM_DILATION_PX * 3, _RIM_DILATION_PX * 3)
    )
    dilated_wide: np.ndarray = cv2.dilate(mask, kernel_wide, iterations=1)
    bg_ring = (dilated_wide > 0) & (dilated == 0)

    bg_ring_pixels = int(np.sum(bg_ring))
    if bg_ring_pixels < _MIN_BOUNDARY_PIXELS:
        # Fallback: compare boundary to overall background
        bg_mask = mask == 0
        if int(np.sum(bg_mask)) == 0:
            return 0.0
        bg_mean = float(np.mean(v_channel[bg_mask]))
    else:
        bg_mean = float(np.mean(v_channel[bg_ring]))

    boundary_mean = float(np.mean(v_channel[boundary]))

    # Rim light: boundary is brighter than nearby background
    differential = boundary_mean - bg_mean

    # Normalize: a differential of 40+ intensity units is strong rim lighting
    # Typical: no rim light ~-10 to 5, subtle ~10-20, strong ~30+
    if differential <= 0:
        return 0.0

    score = min(1.0, differential / 40.0)
    return float(np.clip(score, 0.0, 1.0))


def measure_shadow_quality(v_channel: np.ndarray) -> float:
    """Measure shadow edge softness via gradient magnitude at shadow boundaries.

    Thresholds V to detect shadow regions, then measures gradient magnitude
    at shadow boundaries. Soft gradients indicate carefully designed
    shading; hard edges indicate deliberate cel-shading style.

    Both extremes (very soft, very hard) are scored as intentional/good.
    Mid-range gradients score lower as less distinctive.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).

    Returns
    -------
    float
        Score in [0, 1]. Higher = more distinctive shadow character.
    """
    # Threshold to find shadow regions
    shadow_threshold = int(255 * _SHADOW_V_THRESHOLD)
    shadow_mask = v_channel < shadow_threshold

    shadow_pixels = int(np.sum(shadow_mask))
    total_pixels = v_channel.size
    shadow_ratio = shadow_pixels / total_pixels if total_pixels > 0 else 0.0

    # If very few or no shadows, score low (flat lighting)
    if shadow_ratio < 0.02:
        return 0.1

    # Compute gradient magnitude at shadow boundaries
    v_float = v_channel.astype(np.float32)
    grad_x: np.ndarray = cv2.Sobel(v_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y: np.ndarray = cv2.Sobel(v_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag: np.ndarray = np.sqrt(grad_x**2 + grad_y**2)  # pyright: ignore[reportUnknownArgumentType]

    # Find shadow boundary: dilate shadow mask and XOR to get edge ring
    shadow_u8 = shadow_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated: np.ndarray = cv2.dilate(shadow_u8, kernel, iterations=1)
    eroded: np.ndarray = cv2.erode(shadow_u8, kernel, iterations=1)
    shadow_boundary = (dilated > 0) & (eroded == 0)

    boundary_pixels = int(np.sum(shadow_boundary))
    if boundary_pixels < 20:
        return 0.2

    # Mean gradient at shadow boundaries
    mean_gradient = float(np.mean(grad_mag[shadow_boundary]))

    # Score: both very soft (< 15) and very hard (> 60) shadows are
    # aesthetically intentional. Mid-range is less distinctive.
    # Map via V-shaped curve centered around ~35
    if mean_gradient < 35:
        # Soft shadows: 0 → 1.0, 35 → 0.3
        score = 0.3 + 0.7 * (1.0 - mean_gradient / 35.0)
    else:
        # Hard shadows: 35 → 0.3, 80+ → 1.0
        score = 0.3 + 0.7 * min(1.0, (mean_gradient - 35.0) / 45.0)

    return float(np.clip(score, 0.0, 1.0))


def measure_atmospheric_lighting(v_channel: np.ndarray) -> float:
    """Detect bloom/glow effects by comparing V with Gaussian-blurred V.

    Regions where the original V exceeds the blurred version by a threshold
    indicate bloom/glow sources. Scores based on bloom coverage and intensity.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).

    Returns
    -------
    float
        Score in [0, 1]. Higher = more atmospheric lighting effects.
    """
    v_float = v_channel.astype(np.float32)
    blurred: np.ndarray = cv2.GaussianBlur(v_float, (_BLOOM_KSIZE, _BLOOM_KSIZE), 0)

    # Bloom regions: where original is significantly brighter than blurred
    diff = v_float - blurred
    bloom_mask = diff > _BLOOM_THRESHOLD

    bloom_coverage = float(np.mean(bloom_mask))
    if bloom_coverage < 0.001:
        return 0.0

    # Mean intensity of bloom regions
    bloom_pixels = diff[bloom_mask]
    bloom_intensity = float(np.mean(bloom_pixels)) / 255.0

    # Score combines coverage and intensity
    # Coverage: 5%+ is strong bloom; intensity weights how bright the bloom is
    coverage_score = min(1.0, bloom_coverage / 0.05)
    intensity_score = min(1.0, bloom_intensity * 4.0)

    score = 0.6 * coverage_score + 0.4 * intensity_score
    return float(np.clip(score, 0.0, 1.0))


def measure_highlight_shadow_balance(v_channel: np.ndarray) -> float:
    """Classify tonal balance via tri-zone histogram analysis.

    Divides V into shadows (<0.3), midtones (0.3-0.7), and highlights (>0.7),
    measures their proportions, and scores how purposeful the distribution is.

    Balanced distributions and intentionally skewed distributions (high-key,
    low-key) both score well. Only flat/uniform distributions score low.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).

    Returns
    -------
    float
        Score in [0, 1]. Higher = more purposeful tonal distribution.
    """
    total = v_channel.size
    if total == 0:
        return 0.0

    v_norm = v_channel.astype(np.float32) / 255.0

    shadow_ratio = float(np.mean(v_norm < _SHADOW_V_THRESHOLD))
    highlight_ratio = float(np.mean(v_norm > _HIGHLIGHT_V_THRESHOLD))
    midtone_ratio = 1.0 - shadow_ratio - highlight_ratio

    # Score: reward clear tonal character
    # High-key (highlight_ratio > 0.5): intentional brightness
    # Low-key (shadow_ratio > 0.5): intentional darkness
    # Balanced (midtone_ratio ~0.4-0.6 with some shadows and highlights): good exposure
    # Flat (midtone_ratio > 0.85): no tonal variety

    # Compute entropy of the three-zone distribution as a base
    ratios = np.array([shadow_ratio, midtone_ratio, highlight_ratio])
    ratios = ratios[ratios > 0.001]  # filter near-zero for log stability
    entropy = -float(np.sum(ratios * np.log2(ratios)))
    max_entropy = np.log2(3.0)  # ~1.585 for uniform 3-way split

    # Normalized entropy: 0 = one zone dominates, 1 = perfectly balanced
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # High entropy (balanced) is good. Very low entropy (dominant zone) is also
    # good if it indicates intentional high-key or low-key.
    if norm_entropy > 0.7:
        # Well-balanced — direct correlation with entropy
        score = 0.5 + 0.5 * norm_entropy
    elif shadow_ratio > 0.5 or highlight_ratio > 0.5:
        # Intentional high-key or low-key — strong tonal character
        dominant = max(shadow_ratio, highlight_ratio)
        score = 0.4 + 0.4 * dominant
    else:
        # Neither balanced nor strongly skewed — midtone heavy, less interesting
        score = 0.2 + 0.3 * norm_entropy

    return float(np.clip(score, 0.0, 1.0))


# -- Directionality Classification --


def _classify_directionality(v_channel: np.ndarray) -> str | None:
    """Classify the dominant light direction from 3x3 grid luminance.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).

    Returns
    -------
    str | None
        Direction label (e.g. "top", "left", "top_right") or None if
        no clear direction.
    """
    h, w = v_channel.shape
    grid_h = h // 3
    grid_w = w // 3

    if grid_h == 0 or grid_w == 0:
        return None

    # Build 3x3 grid of mean luminance
    grid = np.zeros((3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            y0 = row * grid_h
            y1 = (row + 1) * grid_h if row < 2 else h
            x0 = col * grid_w
            x1 = (col + 1) * grid_w if col < 2 else w
            grid[row, col] = float(np.mean(v_channel[y0:y1, x0:x1]))

    # Compute directional gradients
    top_mean = float(np.mean(grid[0, :]))
    bottom_mean = float(np.mean(grid[2, :]))
    left_mean = float(np.mean(grid[:, 0]))
    right_mean = float(np.mean(grid[:, 2]))

    overall_mean = float(np.mean(grid))
    if overall_mean < 1.0:
        return None

    # Check each direction: ratio of bright side to dim side
    vert_ratio = top_mean / max(bottom_mean, 1.0)
    horiz_ratio = right_mean / max(left_mean, 1.0)
    vert_ratio_inv = bottom_mean / max(top_mean, 1.0)
    horiz_ratio_inv = left_mean / max(right_mean, 1.0)

    max_ratio = max(vert_ratio, vert_ratio_inv, horiz_ratio, horiz_ratio_inv)
    if max_ratio < _DIRECTIONALITY_THRESHOLD:
        return None

    # Determine dominant direction
    if (
        vert_ratio >= _DIRECTIONALITY_THRESHOLD
        and horiz_ratio >= _DIRECTIONALITY_THRESHOLD
    ):
        return "top_right"
    if (
        vert_ratio >= _DIRECTIONALITY_THRESHOLD
        and horiz_ratio_inv >= _DIRECTIONALITY_THRESHOLD
    ):
        return "top_left"
    if (
        vert_ratio_inv >= _DIRECTIONALITY_THRESHOLD
        and horiz_ratio >= _DIRECTIONALITY_THRESHOLD
    ):
        return "bottom_right"
    if (
        vert_ratio_inv >= _DIRECTIONALITY_THRESHOLD
        and horiz_ratio_inv >= _DIRECTIONALITY_THRESHOLD
    ):
        return "bottom_left"
    if vert_ratio >= _DIRECTIONALITY_THRESHOLD:
        return "top"
    if vert_ratio_inv >= _DIRECTIONALITY_THRESHOLD:
        return "bottom"
    if horiz_ratio >= _DIRECTIONALITY_THRESHOLD:
        return "right"
    if horiz_ratio_inv >= _DIRECTIONALITY_THRESHOLD:
        return "left"

    return None


# -- Score Combination --


def _combine_scores(
    sub_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Combine sub-property scores using weighted mean."""
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in sub_scores.items():
        w = weights.get(name, 0.0)
        weighted_sum += w * score
        total_weight += w

    if total_weight == 0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))


# -- Tag Generation --


def _generate_tags(
    *,
    overall_score: float,
    sub_scores: dict[str, float],
    light_direction: str | None,
    tagger_predictions: dict[str, float] | None,
    confidence_threshold: float,
) -> list[Tag]:
    """Generate lighting tags from analysis results and WD-Tagger predictions.

    Parameters
    ----------
    overall_score : float
        Combined lighting score.
    sub_scores : dict[str, float]
        Per-sub-property scores.
    light_direction : str | None
        Detected dominant light direction, or None.
    tagger_predictions : dict[str, float] | None
        WD-Tagger tag->confidence mapping, or None.
    confidence_threshold : float
        Minimum confidence for emitting a tag.

    Returns
    -------
    list[Tag]
        Lighting tags.
    """
    tags: list[Tag] = []

    # Overall lighting quality
    if overall_score >= 0.7:
        tags.append(
            Tag(name="dramatic_lighting", confidence=overall_score, category="lighting")
        )
    elif overall_score < 0.25:
        tags.append(
            Tag(
                name="flat_lighting",
                confidence=1.0 - overall_score,
                category="lighting",
            )
        )

    # Contrast tags
    contrast = sub_scores.get("contrast_ratio", 0.5)
    if contrast >= 0.7:
        tags.append(Tag(name="high_contrast", confidence=contrast, category="lighting"))
    elif contrast < 0.3:
        tags.append(
            Tag(name="low_contrast", confidence=1.0 - contrast, category="lighting")
        )

    # Rim lighting
    rim = sub_scores.get("rim_edge_lighting", 0.0)
    if rim >= 0.3:
        # Cross-reference with WD-Tagger if available
        tagger_rim = 0.0
        if tagger_predictions is not None:
            tagger_rim = tagger_predictions.get("rim_lighting", 0.0)
        combined_rim = max(rim, (rim + tagger_rim) / 2 if tagger_rim > 0 else rim)
        tags.append(Tag(name="rim_lit", confidence=combined_rim, category="lighting"))

    # Shadow character
    shadow = sub_scores.get("shadow_quality", 0.0)
    if shadow >= 0.6:
        # Determine soft vs hard from the measurement approach
        # shadow_quality scores high for both extremes; check atmospheric
        # as a heuristic — atmospheric scenes tend toward soft shadows
        atmo = sub_scores.get("atmospheric_lighting", 0.0)
        if atmo >= 0.3:
            tags.append(
                Tag(name="soft_shadows", confidence=shadow, category="lighting")
            )
        else:
            tags.append(
                Tag(name="hard_shadows", confidence=shadow, category="lighting")
            )

    # Atmospheric effects
    atmo = sub_scores.get("atmospheric_lighting", 0.0)
    if atmo >= 0.3:
        tags.append(Tag(name="atmospheric", confidence=atmo, category="lighting"))

    # Tonal balance tags
    balance = sub_scores.get("highlight_shadow_balance", 0.0)
    if balance >= 0.8:
        tags.append(
            Tag(name="balanced_exposure", confidence=balance, category="lighting")
        )

    # Light direction tag
    if light_direction is not None:
        directionality = sub_scores.get("light_directionality", 0.0)
        if directionality >= 0.3:
            tags.append(
                Tag(
                    name="directional_light",
                    confidence=directionality,
                    category="lighting",
                )
            )

    # WD-Tagger supplementary tags (pass-through, do not affect score)
    if tagger_predictions is not None:
        for tag_name in _TAGGER_LIGHTING_TAGS:
            confidence = tagger_predictions.get(tag_name, 0.0)
            if confidence >= confidence_threshold:
                # Avoid duplicating the classical rim_lit tag
                if tag_name == "rim_lighting" and any(
                    t.name == "rim_lit" for t in tags
                ):
                    continue
                tags.append(
                    Tag(name=tag_name, confidence=confidence, category="lighting")
                )

    return [t for t in tags if t.confidence >= confidence_threshold]


# -- Tonal Classification Metadata --


def _classify_tonality(v_channel: np.ndarray) -> str:
    """Classify overall tonality as high-key, low-key, or balanced.

    Parameters
    ----------
    v_channel : np.ndarray
        V channel uint8 (H, W).

    Returns
    -------
    str
        One of "high_key", "low_key", or "balanced".
    """
    v_norm = v_channel.astype(np.float32) / 255.0
    shadow_ratio = float(np.mean(v_norm < _SHADOW_V_THRESHOLD))
    highlight_ratio = float(np.mean(v_norm > _HIGHLIGHT_V_THRESHOLD))

    if highlight_ratio > 0.5:
        return "high_key"
    if shadow_ratio > 0.5:
        return "low_key"
    return "balanced"


# -- Analyzer Class --


class LightingAnalyzer:
    """Lighting dimension analyzer — illumination design analysis.

    Measures six sub-properties of illumination quality using classical CV
    on the V (value/brightness) channel. Uses segmentation mask for
    rim-light detection and WD-Tagger predictions for supplementary tags.
    """

    name: str = "lighting"

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        """Analyze lighting properties of an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array with shape (H, W, 3).
        config : AnalyzerConfig
            Per-analyzer configuration.
        shared : SharedModels
            Outputs from shared model inference (uses segmentation_mask
            and tagger_predictions).

        Returns
        -------
        AnalyzerResult
            Lighting dimension score, tags, and sub-property metadata.
        """
        h, w = image.shape[:2]
        v_channel = _extract_v_channel(image)

        # Get segmentation mask for rim-light detection
        mask = _get_binary_mask(shared, h, w)

        # Get tagger predictions for supplementary tags
        tagger_predictions = shared.get("tagger_predictions")

        # Get sub-property weights from config or defaults
        weights = dict(DEFAULT_WEIGHTS)
        config_weights: object = config.params.get("sub_weights")
        if isinstance(config_weights, dict):
            for key, val in config_weights.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(key, str) and isinstance(val, int | float):
                    weights[key] = float(val)

        # Compute all sub-properties
        sub_scores: dict[str, float] = {
            "contrast_ratio": measure_contrast_ratio(v_channel),
            "light_directionality": measure_light_directionality(v_channel),
            "rim_edge_lighting": measure_rim_edge_lighting(v_channel, mask),
            "shadow_quality": measure_shadow_quality(v_channel),
            "atmospheric_lighting": measure_atmospheric_lighting(v_channel),
            "highlight_shadow_balance": measure_highlight_shadow_balance(v_channel),
        }

        # Classify light direction for metadata and tag generation
        light_direction = _classify_directionality(v_channel)

        # Combined score
        score = _combine_scores(sub_scores, weights)

        # Generate tags
        tags = _generate_tags(
            overall_score=score,
            sub_scores=sub_scores,
            light_direction=light_direction,
            tagger_predictions=tagger_predictions,
            confidence_threshold=config.confidence_threshold,
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "sub_scores": sub_scores,
            "weights_used": weights,
            "tonality": _classify_tonality(v_channel),
            "light_direction": light_direction,
            "has_segmentation": mask is not None,
        }

        return AnalyzerResult(
            analyzer=self.name,
            score=score,
            tags=tags,
            metadata=metadata,
        )
