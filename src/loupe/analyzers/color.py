"""Color analyzer — palette design analysis.

Measures chromatic properties across seven sub-properties: Matsuda harmony
scoring, palette cohesion, saturation balance, color contrast, color
temperature consistency, palette diversity, and vivid color (colorfulness).

All perceptual computations use OkLab/OkLCh color space. Palette extraction
uses K-means clustering in OkLab with cluster merging for shading variants.
Fully classical — no model dependencies.

Tags produced:
    harmonic_i, harmonic_v, harmonic_I, harmonic_L, harmonic_T,
    harmonic_Y, harmonic_X, harmonic_N — best-fit Matsuda template
    warm_palette, cool_palette, neutral_palette — temperature character
    vivid, muted — colorfulness character
    monochromatic, limited_palette, diverse_palette — diversity character

Known limitations:
    - Color contrast uses overall luminance contrast without segmentation
      (segmentation-enhanced version requires SharedModels from Phase 3+).
    - Calibration is initial/theoretical — will be refined with real images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import KMeans  # pyright: ignore[reportMissingTypeStubs]

from loupe.analyzers._color_space import (
    srgb_uint8_to_oklab,
    srgb_uint8_to_oklch,
)
from loupe.core.models import AnalyzerResult, Tag

if TYPE_CHECKING:
    from loupe.analyzers.base import AnalyzerConfig, SharedModels

# -- Constants --

# Default sub-property weights from RQ2 §5.7
DEFAULT_WEIGHTS: dict[str, float] = {
    "harmony": 0.25,
    "palette_cohesion": 0.15,
    "saturation_balance": 0.10,
    "color_contrast": 0.15,
    "color_temperature": 0.05,
    "palette_diversity": 0.10,
    "vivid_color": 0.20,
}

# Matsuda template tag names (indexed by template type 0-7)
MATSUDA_TEMPLATE_NAMES = [
    "harmonic_i",
    "harmonic_V",
    "harmonic_I",
    "harmonic_L",
    "harmonic_T",
    "harmonic_Y",
    "harmonic_X",
    "harmonic_N",
]

# Matsuda template sectors: each template is a list of (center_offset, half_width)
# pairs in degrees. Center_offset is relative to the template rotation angle.
# Sector geometry from Cohen-Or et al. (2006).
MATSUDA_TEMPLATES: list[list[tuple[float, float]]] = [
    # Type i: single narrow arc (~18°)
    [(0.0, 9.0)],
    # Type V: single wide arc (~93°)
    [(0.0, 46.5)],
    # Type I: two narrow arcs 180° apart
    [(0.0, 9.0), (180.0, 9.0)],
    # Type L: one narrow + one wide, ~90° apart
    [(0.0, 9.0), (90.0, 46.5)],
    # Type T: split-complementary
    [(0.0, 9.0), (120.0, 9.0), (240.0, 9.0)],
    # Type Y: triadic variant
    [(0.0, 46.5), (180.0, 9.0)],
    # Type X: double-complementary
    [(0.0, 9.0), (90.0, 9.0), (180.0, 9.0), (270.0, 9.0)],
    # Type N: rectangle / tetradic
    [(0.0, 46.5), (180.0, 46.5)],
]

# Downsample target for K-means performance
_DOWNSAMPLE_SIZE = 256

# Default number of K-means clusters
_DEFAULT_K = 6

# OkLab distance threshold for merging similar clusters (shading variants)
_MERGE_THRESHOLD = 0.05

# Minimum chroma in OkLCh to consider a pixel as chromatic
_CHROMA_THRESHOLD = 0.02


# -- Palette Extraction --


def _downsample(image: np.ndarray, target_size: int = _DOWNSAMPLE_SIZE) -> np.ndarray:
    """Downsample image to approximately target_size x target_size via striding."""
    h, w = image.shape[:2]
    if h <= target_size and w <= target_size:
        return image
    factor = max(h, w) / target_size
    step = max(1, int(factor))
    return image[::step, ::step]


def extract_palette(
    oklab_image: np.ndarray,
    n_clusters: int = _DEFAULT_K,
    merge_threshold: float = _MERGE_THRESHOLD,
) -> list[tuple[np.ndarray, float]]:
    """Extract color palette via K-means clustering in OkLab space.

    Parameters
    ----------
    oklab_image : np.ndarray
        OkLab image, shape (H, W, 3), float32.
    n_clusters : int
        Number of initial K-means clusters.
    merge_threshold : float
        OkLab Euclidean distance below which clusters are merged.

    Returns
    -------
    list[tuple[np.ndarray, float]]
        List of (centroid_oklab, pixel_proportion) sorted by proportion descending.
    """
    pixels = oklab_image.reshape(-1, 3)
    n_samples = len(pixels)
    if n_samples == 0:
        return []

    k = min(n_clusters, n_samples)
    kmeans = KMeans(  # pyright: ignore[reportUnknownVariableType]
        n_clusters=k,
        n_init=4,  # pyright: ignore[reportArgumentType]
        max_iter=100,
        random_state=42,
    )
    labels: np.ndarray = np.asarray(kmeans.fit_predict(pixels))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    centroids: np.ndarray = np.asarray(kmeans.cluster_centers_).astype(np.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    # Compute pixel proportions per cluster
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    proportions = counts / n_samples

    # Merge clusters within threshold
    merged_centroids: list[np.ndarray] = []
    merged_proportions: list[float] = []
    used = np.zeros(k, dtype=bool)

    for i in range(k):
        if used[i]:
            continue
        group_indices = [i]
        for j in range(i + 1, k):
            if used[j]:
                continue
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < merge_threshold:
                group_indices.append(j)
                used[j] = True
        used[i] = True

        # Weighted average centroid
        group_props = np.array([proportions[idx] for idx in group_indices])
        total_prop = float(group_props.sum())
        if total_prop > 0:
            weighted_centroid: np.ndarray = (
                np.sum(
                    np.array(
                        [centroids[idx] * proportions[idx] for idx in group_indices]
                    ),
                    axis=0,
                )
                / total_prop
            )
            merged_centroids.append(weighted_centroid.astype(np.float32))
            merged_proportions.append(total_prop)

    # Sort by proportion descending
    order = np.argsort(merged_proportions)[::-1]
    return [
        (merged_centroids[i], merged_proportions[i])
        for i in order
        if merged_proportions[i] > 0
    ]


# -- Matsuda Harmony Scoring --


def _build_hue_histogram(
    oklch_image: np.ndarray,
    n_bins: int = 360,
    chroma_threshold: float = _CHROMA_THRESHOLD,
) -> np.ndarray:
    """Build chroma-weighted hue histogram, excluding low-chroma pixels.

    Parameters
    ----------
    oklch_image : np.ndarray
        OkLCh image, shape (H, W, 3) with channels (L, C, h).
    n_bins : int
        Number of hue bins.
    chroma_threshold : float
        Minimum chroma to include a pixel.

    Returns
    -------
    np.ndarray
        Chroma-weighted hue histogram, shape (n_bins,), float64.
        Normalized so values sum to 1.0 (or all zeros if no chromatic pixels).
    """
    pixels = oklch_image.reshape(-1, 3)
    chroma = pixels[:, 1]
    hue = pixels[:, 2]

    mask = chroma >= chroma_threshold
    if not np.any(mask):
        return np.zeros(n_bins, dtype=np.float64)

    chroma_filtered = chroma[mask].astype(np.float64)
    hue_filtered = hue[mask].astype(np.float64)

    bin_indices = np.clip(
        (hue_filtered / 360.0 * n_bins).astype(np.int64), 0, n_bins - 1
    )
    histogram = np.bincount(bin_indices, weights=chroma_filtered, minlength=n_bins)

    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram


def _angular_distance(h1: np.ndarray, h2: float) -> np.ndarray:
    """Minimum angular distance between hue arrays and a reference hue."""
    diff = np.abs(h1 - h2) % 360.0
    return np.minimum(diff, 360.0 - diff)


def _template_cost(
    hue_bins: np.ndarray,
    histogram: np.ndarray,
    sectors: list[tuple[float, float]],
    rotation: float,
) -> float:
    """Compute harmony cost for a template at a given rotation.

    Cost = sum of (histogram_weight * min_distance_to_any_sector) for each bin.
    """
    n_bins = len(histogram)
    # Distance from each hue bin to the nearest sector
    min_dist = np.full(n_bins, 180.0, dtype=np.float64)
    for center_offset, half_width in sectors:
        sector_center = (rotation + center_offset) % 360.0
        dist = _angular_distance(hue_bins, sector_center)
        # Distance from sector edge: max(0, angular_dist - half_width)
        dist_from_edge = np.maximum(0.0, dist - half_width)
        min_dist = np.minimum(min_dist, dist_from_edge)

    return float(np.sum(histogram * min_dist))


def compute_harmony(oklch_image: np.ndarray) -> tuple[float, int]:
    """Compute Matsuda harmony score via Cohen-Or template fitting.

    Parameters
    ----------
    oklch_image : np.ndarray
        OkLCh image, shape (H, W, 3).

    Returns
    -------
    tuple[float, int]
        (harmony_score in [0, 1], best_template_index).
        Score of 1.0 means perfect harmony; 0.0 means maximum disharmony.
    """
    histogram = _build_hue_histogram(oklch_image)
    if histogram.sum() == 0:
        # No chromatic content — achromatic images are trivially harmonious
        return 1.0, 0

    n_bins = len(histogram)
    hue_bins = np.linspace(0.5, 359.5, n_bins)

    # Maximum theoretical cost: all hue weight at 180° from nearest sector
    # For the widest single sector (type V, half_width 46.5), max distance
    # is 180 - 46.5 = 133.5. Use 180 as conservative upper bound.
    max_cost = 180.0  # theoretical maximum when all weight is opposite sector

    best_cost = float("inf")
    best_template = 0

    for t_idx, sectors in enumerate(MATSUDA_TEMPLATES):
        for rotation in range(360):
            cost = _template_cost(hue_bins, histogram, sectors, float(rotation))
            if cost < best_cost:
                best_cost = cost
                best_template = t_idx

    score = max(0.0, 1.0 - best_cost / max_cost)
    return score, best_template


# -- Sub-Property Measurements --


def measure_palette_cohesion(
    palette: list[tuple[np.ndarray, float]],
) -> float:
    """Measure palette cohesion via pairwise OkLab distance mean and variance.

    Colors that are close together in OkLab = cohesive palette.
    Both the mean distance (are colors similar?) and variance (are distances
    consistent?) contribute to the score.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more cohesive palette.
    """
    if len(palette) < 2:
        return 1.0  # Single color is perfectly cohesive

    centroids = np.array([c for c, _ in palette], dtype=np.float32)
    n = len(centroids)
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    if not distances:
        return 1.0

    mean_dist = float(np.mean(distances))
    variance = float(np.var(distances))

    # Mean distance component: closer colors = more cohesive.
    # OkLab distances above ~0.5 are very different colors.
    mean_score = 1.0 / (1.0 + mean_dist / 0.15)

    # Variance component: consistent spacing = more cohesive.
    var_score = 1.0 / (1.0 + variance / 0.02)

    # Combined: mean distance matters more than variance
    score = 0.7 * mean_score + 0.3 * var_score
    return float(np.clip(score, 0.0, 1.0))


def measure_saturation_balance(oklch_image: np.ndarray) -> float:
    """Measure saturation balance from OkLCh chroma distribution.

    Balanced = low coefficient of variation in chroma + moderate entropy.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more balanced saturation.
    """
    chroma = oklch_image[..., 1].ravel()
    if len(chroma) == 0:
        return 0.0

    mean_c = float(np.mean(chroma))
    std_c = float(np.std(chroma))

    if mean_c < 1e-6:
        # Near-zero chroma everywhere — achromatic, low saturation balance
        return 0.2

    # Coefficient of variation: lower = more balanced
    cv = std_c / mean_c
    # Map CV to score: CV of 0 → 1.0, CV of 2+ → near 0
    cv_score = 1.0 / (1.0 + cv)

    # Chroma histogram entropy (16 bins)
    hist, _ = np.histogram(chroma, bins=16, range=(0, float(np.max(chroma)) + 1e-6))
    hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
    # Entropy (higher = more uniform distribution)
    nonzero = hist_norm[hist_norm > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    max_entropy = np.log2(16)
    entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(np.clip(0.6 * cv_score + 0.4 * entropy_score, 0.0, 1.0))


def measure_color_contrast(oklab_image: np.ndarray) -> float:
    """Measure color contrast via luminance range and chroma spread.

    Without segmentation, uses overall image statistics.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more color contrast.
    """
    l_channel = oklab_image[..., 0].ravel()
    if len(l_channel) == 0:
        return 0.0

    # Luminance contrast: 95th - 5th percentile
    p5 = float(np.percentile(l_channel, 5))
    p95 = float(np.percentile(l_channel, 95))
    lum_contrast = p95 - p5  # OkLab L is in [0, 1]

    # Chroma spread: standard deviation of a and b channels
    a_std = float(np.std(oklab_image[..., 1].ravel()))
    b_std = float(np.std(oklab_image[..., 2].ravel()))
    chroma_spread = np.sqrt(a_std**2 + b_std**2)
    # Normalize chroma spread: values above ~0.1 are high contrast
    chroma_score = min(1.0, chroma_spread / 0.1)

    return float(np.clip(0.6 * lum_contrast + 0.4 * chroma_score, 0.0, 1.0))


def measure_color_temperature(oklch_image: np.ndarray) -> float:
    """Measure color temperature consistency.

    Scores how consistently warm or cool the palette is.
    Mixed warm/cool without purpose → lower score.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more consistent temperature.
    """
    pixels = oklch_image.reshape(-1, 3)
    chroma = pixels[:, 1]
    hue = pixels[:, 2]

    # Only consider chromatic pixels
    mask = chroma >= _CHROMA_THRESHOLD
    if not np.any(mask):
        return 0.5  # Achromatic — neutral temperature

    hue_filtered = hue[mask]

    # Warm hues: roughly 0-60 and 300-360 degrees (reds, oranges, yellows)
    # Cool hues: roughly 120-270 degrees (greens, blues, purples)
    warm_mask = (hue_filtered < 60) | (hue_filtered > 300)
    cool_mask = (hue_filtered > 120) & (hue_filtered < 270)

    total = len(hue_filtered)
    warm_ratio = float(np.sum(warm_mask)) / total
    cool_ratio = float(np.sum(cool_mask)) / total

    # Consistency = how dominant one temperature is
    # If mostly warm or mostly cool → high consistency
    dominance = max(warm_ratio, cool_ratio)
    # Neutral zone (60-120, 270-300 degrees) pixels don't hurt consistency

    # Map: dominance of 1.0 → 1.0, dominance of 0.5 → ~0.3 (evenly split)
    score = dominance**0.5
    return float(np.clip(score, 0.0, 1.0))


def measure_palette_diversity(oklch_image: np.ndarray) -> float:
    """Measure palette diversity via Simpson's diversity index on 12 hue bins.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more diverse hue distribution.
        Monochromatic → near 0, even spread → near 1.
    """
    pixels = oklch_image.reshape(-1, 3)
    chroma = pixels[:, 1]
    hue = pixels[:, 2]

    mask = chroma >= _CHROMA_THRESHOLD
    if not np.any(mask):
        return 0.0  # No chromatic content → zero diversity

    hue_filtered = hue[mask]

    # 12 bins (30° each, like a color wheel with 12 sectors)
    bin_indices = np.clip((hue_filtered / 30.0).astype(np.int64), 0, 11)
    counts = np.bincount(bin_indices, minlength=12).astype(np.float64)
    total = counts.sum()

    if total < 2:
        return 0.0

    # Simpson's diversity index: D = 1 - Σ(p_i²)
    proportions = counts / total
    simpson = 1.0 - float(np.sum(proportions**2))

    # Normalize to [0, 1]: max possible D = 1 - 1/12 ≈ 0.917
    max_simpson = 1.0 - 1.0 / 12.0
    return float(np.clip(simpson / max_simpson, 0.0, 1.0))


def measure_vivid_color(image_rgb: np.ndarray) -> float:
    """Measure colorfulness via Hasler & Süsstrunk metric.

    Operates in sRGB space as specified by the metric.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB uint8 image, shape (H, W, 3).

    Returns
    -------
    float
        Score in [0, 1]. Higher = more vivid/colorful.
    """
    rgb = image_rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    rg = r - g
    yb = 0.5 * (r + g) - b

    sigma_rg = float(np.std(rg))
    sigma_yb = float(np.std(yb))
    mu_rg = float(np.mean(rg))
    mu_yb = float(np.mean(yb))

    colorfulness = np.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * np.sqrt(
        mu_rg**2 + mu_yb**2
    )

    # Normalize: Hasler & Susstrunk values typically range 0-200+ for uint8.
    # Map with sigmoid: 0 → 0, ~70 → ~0.5, ~150+ → ~0.9+
    score = 1.0 - np.exp(-colorfulness / 100.0)
    return float(np.clip(score, 0.0, 1.0))


# -- Tag Generation --


def _generate_tags(
    *,
    harmony_score: float,
    best_template: int,
    temperature_score: float,
    vivid_score: float,
    diversity_score: float,
    oklch_image: np.ndarray,
    confidence_threshold: float,
) -> list[Tag]:
    """Generate color tags from sub-property measurements."""
    tags: list[Tag] = []

    # Harmony template tag
    if harmony_score >= confidence_threshold:
        tags.append(
            Tag(
                name=MATSUDA_TEMPLATE_NAMES[best_template],
                confidence=harmony_score,
                category="color",
            )
        )

    # Temperature tags
    pixels = oklch_image.reshape(-1, 3)
    chroma = pixels[:, 1]
    hue = pixels[:, 2]
    mask = chroma >= _CHROMA_THRESHOLD

    if np.any(mask):
        hue_filtered = hue[mask]
        warm_ratio = float(
            np.sum((hue_filtered < 60) | (hue_filtered > 300)) / len(hue_filtered)
        )
        cool_ratio = float(
            np.sum((hue_filtered > 120) & (hue_filtered < 270)) / len(hue_filtered)
        )

        if warm_ratio > 0.6:
            tags.append(
                Tag(name="warm_palette", confidence=warm_ratio, category="color")
            )
        elif cool_ratio > 0.6:
            tags.append(
                Tag(name="cool_palette", confidence=cool_ratio, category="color")
            )
        elif temperature_score < 0.5:
            tags.append(
                Tag(
                    name="neutral_palette",
                    confidence=1.0 - temperature_score,
                    category="color",
                )
            )

    # Vivid/muted tags
    if vivid_score >= 0.6:
        tags.append(Tag(name="vivid", confidence=vivid_score, category="color"))
    elif vivid_score <= 0.25:
        tags.append(Tag(name="muted", confidence=1.0 - vivid_score, category="color"))

    # Diversity tags
    if diversity_score <= 0.15:
        tags.append(
            Tag(
                name="monochromatic",
                confidence=1.0 - diversity_score,
                category="color",
            )
        )
    elif diversity_score <= 0.35:
        tags.append(
            Tag(
                name="limited_palette",
                confidence=1.0 - diversity_score,
                category="color",
            )
        )
    elif diversity_score >= 0.7:
        tags.append(
            Tag(
                name="diverse_palette",
                confidence=diversity_score,
                category="color",
            )
        )

    # Filter by confidence threshold
    return [t for t in tags if t.confidence >= confidence_threshold]


# -- Score Combination --


def _combine_scores(
    sub_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Combine sub-property scores using weighted mean.

    Parameters
    ----------
    sub_scores : dict[str, float]
        Sub-property name → score.
    weights : dict[str, float]
        Sub-property name → weight.

    Returns
    -------
    float
        Combined score in [0, 1].
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in sub_scores.items():
        w = weights.get(name, 0.0)
        weighted_sum += w * score
        total_weight += w

    if total_weight == 0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))


# -- Analyzer Class --


class ColorAnalyzer:
    """Color dimension analyzer — fully classical, no model dependencies.

    Measures seven sub-properties of color design quality and produces
    a combined score with descriptive tags.
    """

    name: str = "color"

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        """Analyze color properties of an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array with shape (H, W, 3).
        config : AnalyzerConfig
            Per-analyzer configuration.
        shared : SharedModels
            Outputs from shared model inference (unused in Phase 1).

        Returns
        -------
        AnalyzerResult
            Color dimension score, tags, and sub-property metadata.
        """
        # Downsample for K-means performance
        small = _downsample(image)

        # Convert to OkLab for palette extraction
        oklab_small = srgb_uint8_to_oklab(small)

        # Full-resolution conversions for statistics
        oklab_full = srgb_uint8_to_oklab(image)
        oklch_full = srgb_uint8_to_oklch(image)

        # 1. Palette extraction
        n_clusters = config.params.get("n_clusters", _DEFAULT_K)
        palette = extract_palette(oklab_small, n_clusters=n_clusters)

        # 2. Matsuda harmony scoring
        harmony_score, best_template = compute_harmony(oklch_full)

        # 3. Remaining sub-properties
        cohesion_score = measure_palette_cohesion(palette)
        saturation_score = measure_saturation_balance(oklch_full)
        contrast_score = measure_color_contrast(oklab_full)
        temperature_score = measure_color_temperature(oklch_full)
        diversity_score = measure_palette_diversity(oklch_full)
        vivid_score = measure_vivid_color(image)

        # Collect sub-scores
        sub_scores: dict[str, float] = {
            "harmony": harmony_score,
            "palette_cohesion": cohesion_score,
            "saturation_balance": saturation_score,
            "color_contrast": contrast_score,
            "color_temperature": temperature_score,
            "palette_diversity": diversity_score,
            "vivid_color": vivid_score,
        }

        # Get weights from config or use defaults
        weights = dict(DEFAULT_WEIGHTS)
        config_weights: object = config.params.get("sub_weights")
        if isinstance(config_weights, dict):
            for key, val in config_weights.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(key, str) and isinstance(val, int | float):
                    weights[key] = float(val)

        # Combined score
        score = _combine_scores(sub_scores, weights)

        # Generate tags
        tags = _generate_tags(
            harmony_score=harmony_score,
            best_template=best_template,
            temperature_score=temperature_score,
            vivid_score=vivid_score,
            diversity_score=diversity_score,
            oklch_image=oklch_full,
            confidence_threshold=config.confidence_threshold,
        )

        # Build metadata with all sub-property scores and palette info
        palette_info: list[dict[str, Any]] = [
            {"oklab": centroid.tolist(), "proportion": prop}
            for centroid, prop in palette[:8]  # Cap at 8 entries
        ]

        metadata: dict[str, Any] = {
            "sub_scores": sub_scores,
            "weights_used": weights,
            "best_harmony_template": MATSUDA_TEMPLATE_NAMES[best_template],
            "palette": palette_info,
            "n_palette_colors": len(palette),
        }

        return AnalyzerResult(
            analyzer=self.name,
            score=score,
            tags=tags,
            metadata=metadata,
        )
