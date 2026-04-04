"""Detail analyzer — visual complexity analysis.

Measures the rendering effort and visual complexity of an anime frame
across six sub-properties: edge density, spatial frequency, texture
richness, shading granularity, line work quality, and rendering clarity.

Uses the shared segmentation mask from SharedModels to separate
foreground (character) and background regions. Classical CV metrics are
computed per region and combined with configurable weights (default 60%
background / 40% character).

Tags produced:
    high_detail — overall detail score is very high (>0.7)
    rich_background — background region has high detail
    detailed_character — character region has high detail
    sharp_rendering — rendering clarity is high across the frame
    complex_shading — shading granularity indicates many tonal levels
    fine_line_work — line work quality is high (sharp, clean edges)

Known limitations:
    - Motion blur can inflate edge density metrics. The multi-sub-property
      approach dilutes this effect but does not eliminate it.
    - GLCM texture richness is less informative on anime's flat regions.
      It works better on detailed backgrounds (foliage, cityscapes) than
      on character regions. Weighted lower by default.
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

# Default sub-property weights from RQ2 §6.7
DEFAULT_WEIGHTS: dict[str, float] = {
    "edge_density": 0.20,
    "spatial_frequency": 0.15,
    "texture_richness": 0.10,
    "shading_granularity": 0.20,
    "line_work_quality": 0.20,
    "rendering_clarity": 0.15,
}

# Default region weights (background vs character)
_DEFAULT_BG_WEIGHT = 0.6
_DEFAULT_CHAR_WEIGHT = 0.4

# Segmentation mask binarization threshold
_MASK_THRESHOLD = 0.5

# Character region dominance threshold (>90% frame = treat as all character)
_CHAR_DOMINANCE_THRESHOLD = 0.90

# Minimum region pixel count for meaningful measurement
_MIN_REGION_PIXELS = 100

# Downscale target for analysis (long edge)
_ANALYSIS_SIZE = 512

# GLCM parameters
_GLCM_LEVELS = 64
_GLCM_DISTANCES = [1, 3, 5]
_GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# Shading granularity: V-histogram smoothing kernel size
_SHADING_SMOOTH_KERNEL = 5

# Rendering clarity: local patch size for blur detection
_CLARITY_PATCH_SIZE = 32

# Canny thresholds for line work quality
_CANNY_LOW = 50
_CANNY_HIGH = 150


# -- Image Preparation --


def _prepare_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downscale image and return RGB + grayscale at analysis resolution.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rgb, gray) both at analysis resolution.
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge > _ANALYSIS_SIZE:
        scale = _ANALYSIS_SIZE / long_edge
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        rgb: np.ndarray = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
    else:
        rgb = image

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def _get_region_masks(
    shared: SharedModels, h: int, w: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract binary foreground/background masks from shared segmentation.

    Parameters
    ----------
    shared : SharedModels
        Shared model outputs.
    h : int
        Image height at analysis resolution.
    w : int
        Image width at analysis resolution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (fg_mask, bg_mask, char_ratio) where masks are bool arrays and
        char_ratio is the fraction of pixels that are foreground.
    """
    seg_mask = shared.get("segmentation_mask")
    if seg_mask is None:
        # No segmentation: treat entire image as background
        bg_mask = np.ones((h, w), dtype=bool)
        fg_mask = np.zeros((h, w), dtype=bool)
        return fg_mask, bg_mask, 0.0

    # Resize mask to analysis resolution if needed
    if seg_mask.shape[:2] != (h, w):
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    binary = seg_mask > _MASK_THRESHOLD
    char_ratio = float(np.mean(binary))

    fg_mask = binary
    bg_mask = ~binary

    return fg_mask, bg_mask, char_ratio


# -- Sub-Property Measurements --


def measure_edge_density(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Measure edge density via Laplacian variance.

    Higher variance = more edges = more visual detail/complexity.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Optional bool mask to restrict measurement to specific region.

    Returns
    -------
    float
        Score in [0, 1]. Higher = denser edges (more detail).
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is not None and int(np.sum(mask)) >= _MIN_REGION_PIXELS:
        variance = float(np.var(laplacian[mask]))
    else:
        variance = float(np.var(laplacian))

    # Normalize: Laplacian variance of ~2000+ is very high detail for anime
    # Typical range: flat fill ~10-50, moderate detail ~200-800, high detail ~1000+
    score = min(1.0, variance / 2000.0)
    return float(np.clip(score, 0.0, 1.0))


def measure_spatial_frequency(
    gray: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Measure high-frequency energy ratio via FFT.

    Computes the 2D FFT and measures the ratio of energy in high-frequency
    bins to total energy. Complements edge density with a frequency-domain
    perspective.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Optional bool mask. If provided, pixels outside the mask are zeroed
        before FFT (windowed analysis).

    Returns
    -------
    float
        Score in [0, 1]. Higher = more high-frequency content.
    """
    img = gray.astype(np.float32)

    if mask is not None and int(np.sum(mask)) >= _MIN_REGION_PIXELS:
        # Zero out non-region pixels to focus FFT on the region
        img = img * mask.astype(np.float32)
        region_fraction = float(np.mean(mask))
    else:
        region_fraction = 1.0

    if region_fraction < 0.01:
        return 0.0

    # Compute 2D FFT and shift DC to center
    f = np.fft.fft2(img)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted) ** 2

    h, w = gray.shape[:2]
    cy, cx = h // 2, w // 2

    # Create radial distance map from center
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt(
        (y_coords - cy).astype(np.float64) ** 2
        + (x_coords - cx).astype(np.float64) ** 2
    )
    max_radius = np.sqrt(float(cy * cy + cx * cx))

    if max_radius < 1:
        return 0.0

    # High-frequency: outer 60% of frequency space
    high_freq_mask = dist > (max_radius * 0.4)

    total_energy = float(np.sum(magnitude))
    if total_energy < 1e-10:
        return 0.0

    high_freq_energy = float(np.sum(magnitude[high_freq_mask]))
    ratio = high_freq_energy / total_energy

    # Adjust for region fraction (smaller regions have less total energy)
    if region_fraction < 1.0:
        ratio = ratio / region_fraction

    # Normalize: ratio of ~0.3+ indicates high-frequency content
    score = min(1.0, ratio / 0.3)
    return float(np.clip(score, 0.0, 1.0))


def measure_texture_richness(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Measure texture richness via GLCM entropy.

    Computes the Gray-Level Co-occurrence Matrix at multiple distances and
    angles, then measures the average entropy. Higher entropy = richer texture.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Optional bool mask. If provided, only masked pixels contribute.

    Returns
    -------
    float
        Score in [0, 1]. Higher = richer texture variation.
    """
    from skimage.feature import (
        graycomatrix,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
        graycoprops,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    )

    # Quantize to fewer levels for manageable GLCM
    quantized = (gray.astype(np.float32) / 256.0 * _GLCM_LEVELS).astype(np.uint8)
    quantized = np.clip(quantized, 0, _GLCM_LEVELS - 1)

    if mask is not None and int(np.sum(mask)) >= _MIN_REGION_PIXELS:
        # Extract only masked pixels — reshape into a strip for GLCM
        # This loses spatial structure but provides a tractable approximation
        pixels = quantized[mask]
        side = int(np.sqrt(len(pixels)))
        if side < 4:
            return 0.0
        # Reshape to a square-ish patch
        n_pixels = side * side
        patch = pixels[:n_pixels].reshape(side, side)
    else:
        # Downsample for performance
        h, w = gray.shape[:2]
        max_dim = 256
        if max(h, w) > max_dim:
            scale_factor = max_dim / max(h, w)
            new_w = max(4, int(w * scale_factor))
            new_h = max(4, int(h * scale_factor))
            patch = cv2.resize(
                quantized, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            patch = quantized

    # Compute GLCM
    glcm: np.ndarray = graycomatrix(  # pyright: ignore[reportUnknownMemberType]
        patch,
        distances=_GLCM_DISTANCES,
        angles=_GLCM_ANGLES,
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    # Compute entropy from GLCM (not available via graycoprops)
    entropies: list[float] = []
    for d_idx in range(len(_GLCM_DISTANCES)):
        for a_idx in range(len(_GLCM_ANGLES)):
            p = glcm[:, :, d_idx, a_idx]
            p_nonzero = p[p > 0]
            entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
            entropies.append(entropy)

    if not entropies:
        return 0.0

    avg_entropy = float(np.mean(entropies))

    # Supplementary: GLCM contrast
    contrast_vals: np.ndarray = graycoprops(glcm, "contrast")  # pyright: ignore[reportUnknownMemberType]
    avg_contrast = float(np.mean(contrast_vals))

    # Normalize entropy: max theoretical entropy for 64 levels = log2(64) = 6.0
    # Typical anime flat regions: ~2-3, detailed textures: ~4-5
    entropy_score = min(1.0, avg_entropy / 5.0)

    # Normalize contrast: high contrast indicates texture variation
    contrast_score = min(1.0, avg_contrast / 500.0)

    # Combine: entropy is primary, contrast is supplementary
    score = 0.7 * entropy_score + 0.3 * contrast_score
    return float(np.clip(score, 0.0, 1.0))


def measure_shading_granularity(
    gray: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Measure shading granularity via V-histogram analysis.

    Counts distinct tonal modes (peaks in the value histogram) and
    measures entropy of the value distribution. More tones = more
    sophisticated shading/rendering.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W) — used as value channel proxy.
    mask : np.ndarray | None
        Optional bool mask to restrict to a region.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more complex shading with many tonal levels.
    """
    from scipy.signal import (  # pyright: ignore[reportMissingTypeStubs]
        find_peaks,  # pyright: ignore[reportUnknownVariableType]
    )

    if mask is not None and int(np.sum(mask)) >= _MIN_REGION_PIXELS:
        pixels = gray[mask]
    else:
        pixels = gray.ravel()

    if len(pixels) < _MIN_REGION_PIXELS:
        return 0.0

    # Compute histogram (256 bins)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    hist_float = hist.astype(np.float32)

    # Smooth histogram to find meaningful peaks (not noise)
    kernel = np.ones(_SHADING_SMOOTH_KERNEL) / _SHADING_SMOOTH_KERNEL
    smoothed = np.convolve(hist_float, kernel, mode="same")

    # Find peaks — minimum height to avoid noise peaks
    min_height = float(np.max(smoothed)) * 0.05
    peak_indices, _peak_props = find_peaks(smoothed, height=min_height, distance=8)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    n_peaks: int = len(peak_indices)  # pyright: ignore[reportUnknownArgumentType]

    # Peak count score: 2-3 tones is basic anime, 5+ is complex shading
    # 1 peak -> flat, 3 -> basic, 6+ -> rich
    peak_score = min(1.0, max(0.0, (n_peaks - 1) / 6.0))

    # Entropy of normalized histogram
    total = float(np.sum(hist_float))
    if total < 1:
        return peak_score

    p = hist_float / total
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))

    # Max entropy for 256 bins = 8.0; typical anime: 4-6
    entropy_score = min(1.0, entropy / 7.0)

    # Combine: both signals matter
    score = 0.5 * peak_score + 0.5 * entropy_score
    return float(np.clip(score, 0.0, 1.0))


def measure_line_work_quality(
    gray: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Measure line work quality via edge sharpness and continuity.

    Detects edges with Canny, then measures the Sobel gradient magnitude
    at edge pixels (sharpness) and edge pixel density (continuity).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Optional bool mask to restrict to a region.

    Returns
    -------
    float
        Score in [0, 1]. Higher = sharper, cleaner line work.
    """
    edges: np.ndarray = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)

    # Sobel magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag: np.ndarray = np.sqrt(sobel_x**2 + sobel_y**2)  # pyright: ignore[reportUnknownArgumentType]

    if mask is not None and int(np.sum(mask)) >= _MIN_REGION_PIXELS:
        region_edges = edges & mask.astype(np.uint8)
        region_pixels = int(np.sum(mask))
    else:
        region_edges = edges
        region_pixels = gray.size

    edge_count = int(np.sum(region_edges > 0))
    if edge_count < 10 or region_pixels < _MIN_REGION_PIXELS:
        return 0.0

    # Sharpness: mean gradient magnitude at edge pixels
    edge_positions = region_edges > 0
    mean_sharpness = float(np.mean(gradient_mag[edge_positions]))

    # Normalize sharpness: ~80+ is sharp for anime line art
    sharpness_score = min(1.0, mean_sharpness / 80.0)

    # Edge density within region (line work coverage)
    edge_density = edge_count / region_pixels

    # Normalize: anime line art typically has ~2-8% edge density
    density_score = min(1.0, edge_density / 0.08)

    # Combine: sharpness matters more than density
    score = 0.6 * sharpness_score + 0.4 * density_score
    return float(np.clip(score, 0.0, 1.0))


def measure_rendering_clarity(
    gray: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Measure rendering clarity via global + local Laplacian variance.

    Global: overall Laplacian variance (reuses edge density signal).
    Local: patch-wise Laplacian variance to detect blurry regions.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Optional bool mask to restrict to a region.

    Returns
    -------
    float
        Score in [0, 1]. Higher = sharper, clearer rendering.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is not None and int(np.sum(mask)) >= _MIN_REGION_PIXELS:
        global_var = float(np.var(laplacian[mask]))
        active_mask = mask
    else:
        global_var = float(np.var(laplacian))
        active_mask = np.ones(gray.shape[:2], dtype=bool)

    # Global clarity score
    global_score = min(1.0, global_var / 2000.0)

    # Local patch analysis: detect proportion of sharp patches
    h, w = gray.shape[:2]
    patch_size = _CLARITY_PATCH_SIZE
    n_patches_y = h // patch_size
    n_patches_x = w // patch_size

    if n_patches_y < 1 or n_patches_x < 1:
        return float(np.clip(global_score, 0.0, 1.0))

    sharp_patches = 0
    total_patches = 0

    for py in range(n_patches_y):
        for px in range(n_patches_x):
            y0 = py * patch_size
            y1 = y0 + patch_size
            x0 = px * patch_size
            x1 = x0 + patch_size

            patch_mask = active_mask[y0:y1, x0:x1]
            if int(np.sum(patch_mask)) < patch_size:
                continue

            total_patches += 1
            patch_lap = laplacian[y0:y1, x0:x1]
            patch_var = float(np.var(patch_lap[patch_mask]))

            # Threshold: patch variance > 100 is reasonably sharp
            if patch_var > 100.0:
                sharp_patches += 1

    local_score = sharp_patches / total_patches if total_patches > 0 else global_score

    # Combine: global variance and local sharpness proportion
    score = 0.5 * global_score + 0.5 * local_score
    return float(np.clip(score, 0.0, 1.0))


# -- Region-Separated Analysis --


def _measure_region(gray: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    """Compute all sub-property scores for a single region.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).
    mask : np.ndarray | None
        Bool mask for the region (None = entire image).

    Returns
    -------
    dict[str, float]
        Sub-property name -> score mapping.
    """
    return {
        "edge_density": measure_edge_density(gray, mask),
        "spatial_frequency": measure_spatial_frequency(gray, mask),
        "texture_richness": measure_texture_richness(gray, mask),
        "shading_granularity": measure_shading_granularity(gray, mask),
        "line_work_quality": measure_line_work_quality(gray, mask),
        "rendering_clarity": measure_rendering_clarity(gray, mask),
    }


def _combine_sub_scores(
    sub_scores: dict[str, float], weights: dict[str, float]
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
    bg_scores: dict[str, float] | None,
    char_scores: dict[str, float] | None,
    combined_scores: dict[str, float],
    confidence_threshold: float,
) -> list[Tag]:
    """Generate detail tags from analysis results."""
    tags: list[Tag] = []

    # High overall detail
    if overall_score >= 0.7:
        tags.append(
            Tag(name="high_detail", confidence=overall_score, category="detail")
        )

    # Rich background
    if bg_scores is not None:
        bg_avg = float(np.mean(list(bg_scores.values())))
        if bg_avg >= 0.6:
            tags.append(
                Tag(name="rich_background", confidence=bg_avg, category="detail")
            )

    # Detailed character
    if char_scores is not None:
        char_avg = float(np.mean(list(char_scores.values())))
        if char_avg >= 0.6:
            tags.append(
                Tag(name="detailed_character", confidence=char_avg, category="detail")
            )

    # Sharp rendering
    clarity = combined_scores.get("rendering_clarity", 0.0)
    if clarity >= 0.6:
        tags.append(Tag(name="sharp_rendering", confidence=clarity, category="detail"))

    # Complex shading
    shading = combined_scores.get("shading_granularity", 0.0)
    if shading >= 0.6:
        tags.append(Tag(name="complex_shading", confidence=shading, category="detail"))

    # Fine line work
    line_work = combined_scores.get("line_work_quality", 0.0)
    if line_work >= 0.6:
        tags.append(Tag(name="fine_line_work", confidence=line_work, category="detail"))

    return [t for t in tags if t.confidence >= confidence_threshold]


# -- Analyzer Class --


class DetailAnalyzer:
    """Detail dimension analyzer — region-separated visual complexity.

    Measures six sub-properties of visual detail independently for
    background and character regions (when segmentation is available),
    then combines them using configurable region weights.
    """

    name: str = "detail"

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        """Analyze detail properties of an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array with shape (H, W, 3).
        config : AnalyzerConfig
            Per-analyzer configuration.
        shared : SharedModels
            Outputs from shared model inference (uses segmentation_mask).

        Returns
        -------
        AnalyzerResult
            Detail dimension score, tags, and sub-property metadata.
        """
        # Prepare at analysis resolution
        _rgb, gray = _prepare_image(image)
        h, w = gray.shape[:2]

        # Get region masks
        fg_mask, bg_mask, char_ratio = _get_region_masks(shared, h, w)

        # Get region weights from config
        bg_weight = float(config.params.get("bg_weight", _DEFAULT_BG_WEIGHT))
        char_weight = float(config.params.get("char_weight", _DEFAULT_CHAR_WEIGHT))

        # Get sub-property weights from config or defaults
        sub_weights = dict(DEFAULT_WEIGHTS)
        config_sub_weights: object = config.params.get("sub_weights")
        if isinstance(config_sub_weights, dict):
            for key, val in config_sub_weights.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(key, str) and isinstance(val, int | float):
                    sub_weights[key] = float(val)

        # Determine region scoring strategy based on segmentation availability
        bg_scores: dict[str, float] | None = None
        char_scores: dict[str, float] | None = None

        bg_pixel_count = int(np.sum(bg_mask))
        fg_pixel_count = int(np.sum(fg_mask))

        if char_ratio >= _CHAR_DOMINANCE_THRESHOLD:
            # Character fills >90% of frame: 100% character score
            char_scores = _measure_region(gray, fg_mask)
            combined_scores = dict(char_scores)
            effective_bg_weight = 0.0
            effective_char_weight = 1.0
        elif char_ratio < 0.01 or fg_pixel_count < _MIN_REGION_PIXELS:
            # No character detected: 100% background score
            bg_scores = _measure_region(
                gray, bg_mask if bg_pixel_count >= _MIN_REGION_PIXELS else None
            )
            combined_scores = dict(bg_scores)
            effective_bg_weight = 1.0
            effective_char_weight = 0.0
        else:
            # Both regions present: weighted combination
            bg_scores = _measure_region(gray, bg_mask)
            char_scores = _measure_region(gray, fg_mask)
            combined_scores = {}
            for key in DEFAULT_WEIGHTS:
                bg_val = bg_scores.get(key, 0.0)
                char_val = char_scores.get(key, 0.0)
                combined_scores[key] = bg_weight * bg_val + char_weight * char_val
            effective_bg_weight = bg_weight
            effective_char_weight = char_weight

        # Compute overall score from combined sub-properties
        score = _combine_sub_scores(combined_scores, sub_weights)

        # Generate tags
        tags = _generate_tags(
            overall_score=score,
            bg_scores=bg_scores,
            char_scores=char_scores,
            combined_scores=combined_scores,
            confidence_threshold=config.confidence_threshold,
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "sub_scores": combined_scores,
            "weights_used": sub_weights,
            "region_weights": {
                "background": effective_bg_weight,
                "character": effective_char_weight,
            },
            "character_ratio": round(char_ratio, 4),
            "analysis_resolution": [h, w],
        }

        if bg_scores is not None:
            metadata["bg_sub_scores"] = bg_scores
        if char_scores is not None:
            metadata["char_sub_scores"] = char_scores

        return AnalyzerResult(
            analyzer=self.name,
            score=score,
            tags=tags,
            metadata=metadata,
        )
