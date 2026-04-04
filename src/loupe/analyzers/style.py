"""Style analyzer — artistic identity analysis.

Measures artistic identity through two scored sub-properties and
categorical style tags. This is the least mature analyzer — the
tagging component is solid, but the scoring component is a best-available
proxy rather than a direct measurement.

**Scored sub-properties:**

1. **Aesthetic quality** (weight 0.70): Uses the deepghs/anime_aesthetic
   scorer as a quality proxy. This reflects Danbooru quality ratings that
   correlate with rendering quality, detail, and (partially) style
   consistency, but is not a pure coherence measure.
2. **Layer consistency** (weight 0.30, experimental): Segments the frame
   into foreground/background and measures how consistent the rendering
   properties are *within* each layer. Only intra-layer consistency is
   measured — cross-layer differences are expected in anime and are not
   penalized. Uses edge density uniformity, gradient smoothness, and
   palette coherence per region.

**Style tags (categorical — do not affect score):**

WD-Tagger style tags (passed through when confident):
    flat_color, gradient, realistic, sketch, watercolor_(medium),
    cel_shading, soft_shading, hard_shadow, chromatic_aberration,
    bloom, detailed, simple_background

CLIP zero-shot style categories:
    naturalistic_anime, geometric_abstract_anime, painterly_anime,
    digital_modern_anime, retro_cel_anime

Aesthetic tier tags (from scorer):
    aesthetic_masterpiece, aesthetic_best, aesthetic_great,
    aesthetic_good, aesthetic_normal, aesthetic_low, aesthetic_worst

Known limitations:
    - The aesthetic quality score is entangled with other quality aspects
      (detail, color appeal, composition) — it is a proxy, not a pure
      style coherence measure.
    - Layer consistency is experimental and unvalidated. Its contribution
      is weighted low (0.30) to limit impact from potential false signals.
    - Cross-layer consistency is deliberately ignored, which means
      genuinely incoherent rendering that spans both layers may not be
      detected.
    - CLIP zero-shot categories are broad and may not capture fine-grained
      style distinctions within anime sub-genres.
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

# Default sub-property weights from RQ2 §9.8
DEFAULT_WEIGHTS: dict[str, float] = {
    "aesthetic_quality": 0.70,
    "layer_consistency": 0.30,
}

# Segmentation mask binarization threshold
_MASK_THRESHOLD = 0.5

# Minimum foreground fraction for separate layer analysis
_MIN_REGION_FRACTION = 0.01

# Maximum foreground fraction — above this, treat as single layer
_MAX_REGION_FRACTION = 0.90

# Patch grid size for edge density uniformity measurement
_PATCH_GRID = 4

# Minimum pixels in a region for meaningful measurement
_MIN_REGION_PIXELS = 100

# WD-Tagger style-relevant tag names
_TAGGER_STYLE_TAGS: set[str] = {
    "flat_color",
    "gradient",
    "realistic",
    "sketch",
    "watercolor_(medium)",
    "cel_shading",
    "soft_shading",
    "hard_shadow",
    "chromatic_aberration",
    "bloom",
    "detailed",
    "simple_background",
}

# CLIP zero-shot style category labels
_CLIP_STYLE_LABELS: list[str] = [
    "naturalistic anime",
    "geometric abstract anime",
    "painterly anime",
    "digital modern anime",
    "retro cel anime",
]

# Aesthetic tiers that produce tags (with minimum probability)
_AESTHETIC_TAG_THRESHOLD = 0.3


# -- Image Preparation --


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


def measure_aesthetic_quality(
    aesthetic_prediction: tuple[float, str, dict[str, float]] | None,
) -> float:
    """Extract the continuous aesthetic quality score from the scorer output.

    Parameters
    ----------
    aesthetic_prediction : tuple[float, str, dict[str, float]] | None
        Output from AnimeAestheticScorer.predict(), or None if unavailable.

    Returns
    -------
    float
        Score in [0, 1]. Higher = higher aesthetic quality.
    """
    if aesthetic_prediction is None:
        return 0.5  # Neutral fallback when scorer is unavailable
    return float(np.clip(aesthetic_prediction[0], 0.0, 1.0))


def _measure_edge_uniformity(
    gray: np.ndarray,
    mask: np.ndarray | None,
) -> float:
    """Measure how uniform edge density is across patches within a region.

    Divides the image into a grid, computes edge density per patch
    (restricted to the masked region), and measures consistency as
    1 - normalized_std. High uniformity = consistent detail level.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Binary mask (H, W). None for whole image.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more uniform edge density.
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    patch_h = h // _PATCH_GRID
    patch_w = w // _PATCH_GRID

    if patch_h < 4 or patch_w < 4:
        return 0.5

    densities: list[float] = []
    for row in range(_PATCH_GRID):
        for col in range(_PATCH_GRID):
            y0 = row * patch_h
            y1 = (row + 1) * patch_h if row < _PATCH_GRID - 1 else h
            x0 = col * patch_w
            x1 = (col + 1) * patch_w if col < _PATCH_GRID - 1 else w

            patch_edges = edges[y0:y1, x0:x1]

            if mask is not None:
                patch_mask = mask[y0:y1, x0:x1]
                region_pixels = int(np.sum(patch_mask))
                if region_pixels < _MIN_REGION_PIXELS:
                    continue
                edge_count = int(np.sum(patch_edges[patch_mask > 0]))
                densities.append(edge_count / region_pixels)
            else:
                total = patch_edges.size
                if total == 0:
                    continue
                densities.append(float(np.sum(patch_edges > 0)) / total)

    if len(densities) < 2:
        return 0.5

    arr = np.array(densities)
    mean_density = float(np.mean(arr))
    if mean_density < 1e-6:
        return 1.0  # No edges at all — perfectly uniform (trivially)

    # Coefficient of variation — lower = more uniform
    cv = float(np.std(arr)) / mean_density
    # Map CV to [0, 1]: CV of 0 = 1.0, CV of 1.5+ = 0.0
    score = max(0.0, 1.0 - cv / 1.5)
    return float(np.clip(score, 0.0, 1.0))


def _measure_gradient_consistency(
    gray: np.ndarray,
    mask: np.ndarray | None,
) -> float:
    """Measure consistency of gradient/shading approach within a region.

    Computes local gradient magnitudes and measures how concentrated the
    distribution is. A region with consistently smooth shading OR
    consistently hard cel edges will score high. Mixed approaches
    (some smooth, some hard) score lower.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Binary mask (H, W). None for whole image.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more consistent gradient style.
    """
    # Compute gradient magnitude via Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude: np.ndarray = np.sqrt(grad_x**2 + grad_y**2)  # pyright: ignore[reportUnknownArgumentType]

    if mask is not None:
        region_mask = mask > 0
        if int(np.sum(region_mask)) < _MIN_REGION_PIXELS:
            return 0.5
        values = magnitude[region_mask]
    else:
        values = magnitude.ravel()

    if values.size == 0:
        return 0.5

    # Normalize magnitudes to [0, 1]
    max_val = float(np.max(values))
    if max_val < 1e-6:
        return 1.0  # Flat region — perfectly consistent

    normalized = values / max_val

    # Measure bimodality/concentration via histogram entropy
    # Low entropy = concentrated distribution = consistent approach
    hist, _ = np.histogram(normalized, bins=20, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    probs = hist / total
    # Shannon entropy (max for 20 bins = log2(20) ≈ 4.32)
    entropy = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
    max_entropy = np.log2(20)

    # Lower entropy = more consistent = higher score
    score = 1.0 - (entropy / max_entropy)
    return float(np.clip(score, 0.0, 1.0))


def _measure_palette_coherence(
    image: np.ndarray,
    mask: np.ndarray | None,
) -> float:
    """Measure how well a region's colors fit a small palette.

    Uses K-means with k=4 to extract dominant colors within the region,
    then measures the mean color distance from each pixel to its nearest
    cluster center. Low distance = coherent palette usage.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    mask : np.ndarray | None
        Binary mask (H, W). None for whole image.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more coherent palette.
    """
    if mask is not None:
        region_mask = mask > 0
        pixel_count = int(np.sum(region_mask))
        if pixel_count < _MIN_REGION_PIXELS:
            return 0.5
        pixels = image[region_mask].astype(np.float32)
    else:
        pixels = image.reshape(-1, 3).astype(np.float32)

    # Subsample if too many pixels (K-means can be slow)
    max_samples = 5000
    if len(pixels) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pixels), max_samples, replace=False)
        pixels = pixels[indices]

    # K-means clustering with k=4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = min(4, len(pixels))
    if k < 2:
        return 0.5

    _, labels, centers = cv2.kmeans(  # pyright: ignore[reportUnknownVariableType]
        pixels,
        k,
        None,  # type: ignore[arg-type]
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    # Mean distance from each pixel to its assigned center
    assigned_centers: np.ndarray = centers[labels.ravel()]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    distances = np.sqrt(np.sum((pixels - assigned_centers) ** 2, axis=1))  # pyright: ignore[reportUnknownArgumentType]
    mean_distance = float(np.mean(distances))  # pyright: ignore[reportUnknownArgumentType]

    # Normalize: distance of 0 = perfect coherence, ~80+ = very incoherent
    # (in RGB space, max distance ≈ 441 for black-white, typical anime ~30-60)
    score = max(0.0, 1.0 - mean_distance / 80.0)
    return float(np.clip(score, 0.0, 1.0))


def _measure_region_consistency(
    image: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Measure rendering consistency within a single region.

    Combines edge density uniformity, gradient consistency, and palette
    coherence into a single consistency score.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Binary mask for the region. None for whole image.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more consistent rendering.
    """
    edge_uniformity = _measure_edge_uniformity(gray, mask)
    gradient_consistency = _measure_gradient_consistency(gray, mask)
    palette_coherence = _measure_palette_coherence(image, mask)

    # Equal weighting of the three consistency signals
    return (edge_uniformity + gradient_consistency + palette_coherence) / 3.0


def measure_layer_consistency(
    image: np.ndarray,
    mask: np.ndarray | None,
) -> float:
    """Measure rendering consistency within foreground and background layers.

    Segments the image using the provided mask and measures how consistent
    the rendering properties are *within* each layer independently.
    Cross-layer differences are NOT penalized — anime routinely uses
    different rendering approaches for characters and backgrounds.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    mask : np.ndarray | None
        Binary character mask (H, W) uint8. None if unavailable.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more consistent intra-layer rendering.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if mask is None:
        # No segmentation — measure whole-image consistency
        return _measure_region_consistency(image, gray)

    fg_ratio = float(np.mean(mask > 0))

    # Edge cases: region too small for meaningful separation
    if fg_ratio < _MIN_REGION_FRACTION or fg_ratio > _MAX_REGION_FRACTION:
        return _measure_region_consistency(image, gray)

    # Measure consistency within each layer independently
    fg_mask = (mask > 0).astype(np.uint8)
    bg_mask = (mask == 0).astype(np.uint8)

    fg_consistency = _measure_region_consistency(image, gray, fg_mask)
    bg_consistency = _measure_region_consistency(image, gray, bg_mask)

    # Weight by region proportion
    score = fg_ratio * fg_consistency + (1.0 - fg_ratio) * bg_consistency
    return float(np.clip(score, 0.0, 1.0))


# -- Score Combination --


def _combine_scores(
    sub_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Combine sub-property scores using weighted arithmetic mean.

    Parameters
    ----------
    sub_scores : dict[str, float]
        Per-sub-property scores.
    weights : dict[str, float]
        Per-sub-property weights.

    Returns
    -------
    float
        Combined score in [0, 1].
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for key, score in sub_scores.items():
        w = weights.get(key, 0.0)
        weighted_sum += w * score
        total_weight += w

    if total_weight == 0.0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))


# -- Tag Generation --


def _generate_tags(
    *,
    aesthetic_prediction: tuple[float, str, dict[str, float]] | None,
    tagger_predictions: dict[str, float] | None,
    clip_style_scores: dict[str, float] | None,
    layer_consistency: float,
    confidence_threshold: float,
) -> list[Tag]:
    """Generate style tags from analysis results.

    Tags are categorical context — they do not affect the numerical score.

    Parameters
    ----------
    aesthetic_prediction : tuple[float, str, dict[str, float]] | None
        (score, tier, tier_probabilities) from aesthetic scorer.
    tagger_predictions : dict[str, float] | None
        WD-Tagger tag->confidence mapping, or None.
    clip_style_scores : dict[str, float] | None
        CLIP zero-shot style category probabilities, or None.
    layer_consistency : float
        Layer consistency sub-score.
    confidence_threshold : float
        Minimum confidence for emitting a tag.

    Returns
    -------
    list[Tag]
        Style tags.
    """
    tags: list[Tag] = []

    # Aesthetic tier tags
    if aesthetic_prediction is not None:
        _, _tier, tier_probs = aesthetic_prediction
        for tier_name, prob in tier_probs.items():
            if prob >= _AESTHETIC_TAG_THRESHOLD:
                tag_name = f"aesthetic_{tier_name}"
                tags.append(Tag(name=tag_name, confidence=prob, category="style"))

    # WD-Tagger style tags (pass-through)
    if tagger_predictions is not None:
        for tag_name in _TAGGER_STYLE_TAGS:
            confidence = tagger_predictions.get(tag_name, 0.0)
            if confidence >= confidence_threshold:
                tags.append(Tag(name=tag_name, confidence=confidence, category="style"))

    # CLIP zero-shot style categories
    if clip_style_scores is not None:
        for label, prob in clip_style_scores.items():
            if prob >= confidence_threshold:
                # Convert to tag name: "naturalistic anime" -> "naturalistic_anime"
                tag_name = label.replace(" ", "_")
                tags.append(Tag(name=tag_name, confidence=prob, category="style"))

    # Layer consistency tag (experimental flag)
    if layer_consistency >= 0.7:
        tags.append(
            Tag(
                name="consistent_rendering",
                confidence=layer_consistency,
                category="style",
            )
        )
    elif layer_consistency < 0.3:
        tags.append(
            Tag(
                name="inconsistent_rendering",
                confidence=1.0 - layer_consistency,
                category="style",
            )
        )

    return [t for t in tags if t.confidence >= confidence_threshold]


# -- Analyzer Class --


class StyleAnalyzer:
    """Style dimension analyzer — artistic identity analysis.

    Produces style tags (WD-Tagger + CLIP zero-shot) and a numerical
    score combining aesthetic quality (proxy via anime aesthetic scorer)
    and experimental layer consistency (classical CV).

    This is the least mature analyzer. The tagging component is solid,
    but the scoring component uses a quality proxy rather than a direct
    style coherence measurement.
    """

    name: str = "style"

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        """Analyze style properties of an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array with shape (H, W, 3).
        config : AnalyzerConfig
            Per-analyzer configuration.
        shared : SharedModels
            Outputs from shared model inference (uses aesthetic_prediction,
            tagger_predictions, clip_embedding, and segmentation_mask).

        Returns
        -------
        AnalyzerResult
            Style dimension score, tags, and sub-property metadata.
        """
        h, w = image.shape[:2]

        # Get shared model outputs
        aesthetic_prediction = shared.get("aesthetic_prediction")
        tagger_predictions = shared.get("tagger_predictions")
        mask = _get_binary_mask(shared, h, w)

        # CLIP zero-shot style classification (pre-computed by engine)
        clip_style_scores = shared.get("clip_style_scores")

        # Get sub-property weights from config or defaults
        weights = dict(DEFAULT_WEIGHTS)
        # Support named params from config/default.toml
        aesthetic_w: object = config.params.get("aesthetic_weight")
        if isinstance(aesthetic_w, int | float):
            weights["aesthetic_quality"] = float(aesthetic_w)
        layer_w: object = config.params.get("layer_consistency_weight")
        if isinstance(layer_w, int | float):
            weights["layer_consistency"] = float(layer_w)
        # Also support generic sub_weights override
        config_weights: object = config.params.get("sub_weights")
        if isinstance(config_weights, dict):
            for key, val in config_weights.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(key, str) and isinstance(val, int | float):
                    weights[key] = float(val)

        # Compute sub-property scores
        sub_scores: dict[str, float] = {
            "aesthetic_quality": measure_aesthetic_quality(aesthetic_prediction),
            "layer_consistency": measure_layer_consistency(image, mask),
        }

        # Combined score
        score = _combine_scores(sub_scores, weights)

        # Generate tags
        tags = _generate_tags(
            aesthetic_prediction=aesthetic_prediction,
            tagger_predictions=tagger_predictions,
            clip_style_scores=clip_style_scores,
            layer_consistency=sub_scores["layer_consistency"],
            confidence_threshold=config.confidence_threshold,
        )

        # Build metadata
        aesthetic_tier: str | None = None
        aesthetic_tier_probs: dict[str, float] | None = None
        if aesthetic_prediction is not None:
            _, aesthetic_tier, aesthetic_tier_probs = aesthetic_prediction

        metadata: dict[str, Any] = {
            "sub_scores": sub_scores,
            "weights_used": weights,
            "aesthetic_tier": aesthetic_tier,
            "aesthetic_tier_probabilities": aesthetic_tier_probs,
            "has_segmentation": mask is not None,
            "layer_consistency_experimental": True,
        }

        return AnalyzerResult(
            analyzer=self.name,
            score=score,
            tags=tags,
            metadata=metadata,
        )
