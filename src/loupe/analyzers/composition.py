# Copyright 2025 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Composition analyzer — spatial arrangement analysis.

Measures the structural design of visual elements within an anime frame
across eight sub-properties: rule of thirds, visual balance, symmetry,
depth layering, leading lines, diagonal dominance, negative space, and
framing.

All computations are classical (OpenCV + NumPy + SciPy) with no model
dependencies. An edge-density saliency proxy exploits anime's flat-shading
aesthetic (high-edge subjects against low-edge backgrounds) to approximate
visual attention without a learned saliency model.

Tags produced:
    rule_of_thirds — subject placed near third-line intersections
    centered — subject placed near frame center
    balanced — visual weight evenly distributed
    symmetric — bilateral symmetry detected
    strong_leading_lines — prominent converging lines
    diagonal_composition — strong diagonal line structure
    open_composition — significant negative space
    framed_subject — in-frame elements border the subject

Known limitations:
    - Saliency is approximated via edge density, not a learned model.
      Works well for anime's clean line art but may underperform on
      scenes with uniformly busy or uniformly flat edge distributions.
    - Depth layering is the weakest sub-property due to anime's
      frequent uniform focus across depth planes.
    - Calibration is initial/theoretical — will be refined with real images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from scipy.ndimage import (  # pyright: ignore[reportMissingTypeStubs]
    center_of_mass,  # pyright: ignore[reportUnknownVariableType]
)

from loupe.core.models import AnalyzerResult, Tag

if TYPE_CHECKING:
    from loupe.analyzers.base import AnalyzerConfig, SharedModels

# -- Constants --

# Default sub-property weights from RQ2 §4.6
DEFAULT_WEIGHTS: dict[str, float] = {
    "rule_of_thirds": 0.20,
    "visual_balance": 0.20,
    "symmetry": 0.10,
    "depth_layering": 0.05,
    "leading_lines": 0.15,
    "diagonal_dominance": 0.10,
    "negative_space": 0.15,
    "framing": 0.05,
}

# Downscale target for performance (long edge)
_ANALYSIS_SIZE = 512

# Canny edge detection thresholds (tuned for anime line art)
_CANNY_LOW = 50
_CANNY_HIGH = 150

# Gaussian sigma for saliency center-bias prior (fraction of diagonal)
_CENTER_BIAS_SIGMA = 0.25

# Box filter kernel size for local edge density (fraction of image size)
_DENSITY_KERNEL_FRAC = 0.05

# Rule of thirds: Gaussian sigma as fraction of image diagonal
_THIRDS_SIGMA_FRAC = 0.08

# Negative space: edge density threshold for "empty" blocks
_NEGATIVE_SPACE_THRESHOLD = 0.05

# Framing: outer border fraction of image dimensions
_FRAMING_BORDER_FRAC = 0.15

# Leading lines: minimum line length as fraction of image diagonal
_MIN_LINE_LENGTH_FRAC = 0.08

# Diagonal dominance: angle range for diagonal classification (degrees)
_DIAGONAL_LOW = 30.0
_DIAGONAL_HIGH = 60.0


# -- Image Preparation --


def _prepare_gray(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale image to analysis size and convert to grayscale.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 array (H, W, 3).

    Returns
    -------
    tuple[np.ndarray, float]
        (grayscale uint8 image, scale_factor used for downscaling).
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge > _ANALYSIS_SIZE:
        scale = _ANALYSIS_SIZE / long_edge
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        resized = image
    gray: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return gray, scale


# -- Saliency Proxy --


def compute_saliency(gray: np.ndarray) -> np.ndarray:
    """Compute edge-density saliency map with center-bias prior.

    Canny edge detection -> box filter for local edge density ->
    multiply by center-bias Gaussian.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).

    Returns
    -------
    np.ndarray
        Saliency map (H, W) float32, values in [0, 1].
    """
    h, w = gray.shape[:2]

    # Canny edge detection
    edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)

    # Local edge density via box filter
    kernel_size = max(3, int(max(h, w) * _DENSITY_KERNEL_FRAC) | 1)  # ensure odd
    density: np.ndarray = cv2.boxFilter(
        edges.astype(np.float32), ddepth=-1, ksize=(kernel_size, kernel_size)
    )

    # Normalize to [0, 1]
    d_max = density.max()
    if d_max > 0:
        density = density / d_max

    # Center-bias Gaussian prior
    diag = np.sqrt(float(h * h + w * w))
    sigma = diag * _CENTER_BIAS_SIGMA
    cy, cx = h / 2.0, w / 2.0
    y_coords = np.arange(h, dtype=np.float32) - cy
    x_coords = np.arange(w, dtype=np.float32) - cx
    yy: np.ndarray = np.zeros((h, w), dtype=np.float32)
    xx: np.ndarray = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")  # pyright: ignore[reportUnknownVariableType]
    gaussian: np.ndarray = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))  # pyright: ignore[reportUnknownArgumentType]

    saliency: np.ndarray = density * gaussian

    # Final normalization
    s_max = saliency.max()
    if s_max > 0:
        saliency = saliency / s_max

    return saliency.astype(np.float32)


# -- Sub-Property Measurements --


def measure_rule_of_thirds(saliency: np.ndarray) -> float:
    """Measure saliency proximity to rule-of-thirds power points.

    Places 2D Gaussians at the 4 grid intersections and computes
    the saliency-weighted overlap.

    Returns
    -------
    float
        Score in [0, 1]. Higher = subject near power points.
    """
    h, w = saliency.shape[:2]
    if h < 3 or w < 3:
        return 0.0

    diag = np.sqrt(float(h * h + w * w))
    sigma = diag * _THIRDS_SIGMA_FRAC

    # Four power points at thirds intersections
    points = [
        (h / 3.0, w / 3.0),
        (h / 3.0, 2 * w / 3.0),
        (2 * h / 3.0, w / 3.0),
        (2 * h / 3.0, 2 * w / 3.0),
    ]

    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    yy: np.ndarray = np.zeros((h, w), dtype=np.float32)
    xx: np.ndarray = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")  # pyright: ignore[reportUnknownVariableType]

    # Combined Gaussian field: max over all 4 power points
    combined = np.zeros((h, w), dtype=np.float32)
    for py, px in points:
        g = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma**2))
        combined = np.maximum(combined, g)

    # Saliency-weighted score
    s_sum = saliency.sum()
    if s_sum < 1e-6:
        return 0.0

    score = float(np.sum(saliency * combined) / s_sum)
    return float(np.clip(score, 0.0, 1.0))


def measure_visual_balance(saliency: np.ndarray) -> float:
    """Measure visual balance via center-of-mass deviation and quadrant variance.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more balanced distribution.
    """
    h, w = saliency.shape[:2]
    if h < 2 or w < 2:
        return 0.0

    s_sum = float(saliency.sum())
    if s_sum < 1e-6:
        return 0.5  # Uniform/empty — neutral balance

    # Center of mass deviation from frame center
    com: tuple[float, float] = center_of_mass(saliency)  # pyright: ignore[reportAssignmentType]
    cy, cx = h / 2.0, w / 2.0
    # Normalize deviation by half-dimensions so max deviation = 1.0
    y_dev = abs(com[0] - cy) / cy
    x_dev = abs(com[1] - cx) / cx
    deviation = np.sqrt(y_dev**2 + x_dev**2) / np.sqrt(2.0)  # normalize to [0,1]
    com_score = 1.0 - float(deviation)

    # Quadrant energy variance (lower variance = more balanced)
    mid_y, mid_x = h // 2, w // 2
    quadrants = [
        saliency[:mid_y, :mid_x],
        saliency[:mid_y, mid_x:],
        saliency[mid_y:, :mid_x],
        saliency[mid_y:, mid_x:],
    ]
    energies = [float(q.sum()) for q in quadrants]
    total_energy = sum(energies)
    if total_energy > 0:
        proportions = [e / total_energy for e in energies]
        variance = float(np.var(proportions))
        # Max variance for 4 quadrants is 0.1875 (all in one quadrant)
        var_score = 1.0 - variance / 0.1875
    else:
        var_score = 0.5

    return float(np.clip(0.5 * com_score + 0.5 * var_score, 0.0, 1.0))


def measure_symmetry(edges: np.ndarray) -> float:
    """Measure bilateral symmetry via horizontal flip-and-compare on edge map.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more symmetric.
    """
    h, w = edges.shape[:2]
    if h < 2 or w < 2:
        return 0.0

    # Binarize and dilate edges for tolerance (single-pixel edges rarely
    # align exactly after flip, so a small dilation avoids false negatives)
    edge_binary = (edges > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_dilated: np.ndarray = cv2.dilate(edge_binary, kernel, iterations=1)
    flipped: np.ndarray = cv2.flip(edge_dilated, 1)  # horizontal flip

    edge_f = edge_dilated.astype(np.float32)
    flip_f = flipped.astype(np.float32)

    # IoU of dilated edge pixels
    intersection = float(np.sum(edge_f * flip_f))
    union = float(np.sum(np.clip(edge_f + flip_f, 0, 1)))

    if union < 1.0:
        return 0.0

    iou = intersection / union

    # Raw IoU is typically low even for symmetric images due to pixel
    # misalignment. Apply a scaling to spread the useful range.
    # IoU > 0.3 is quite symmetric for edge maps; IoU > 0.5 is very symmetric.
    score = min(1.0, iou / 0.4)
    return float(np.clip(score, 0.0, 1.0))


def measure_depth_layering(gray: np.ndarray) -> float:
    """Measure depth layering via Laplacian variance across horizontal strips.

    Higher variation in sharpness across strips suggests depth separation
    (e.g., blurred background vs. sharp foreground).

    Returns
    -------
    float
        Score in [0, 1]. Higher = more depth variation.
    """
    h, w = gray.shape[:2]
    n_strips = 5
    strip_h = h // n_strips
    if strip_h < 4 or w < 4:
        return 0.0

    variances: list[float] = []
    for i in range(n_strips):
        y_start = i * strip_h
        y_end = y_start + strip_h
        strip = gray[y_start:y_end, :]
        lap = cv2.Laplacian(strip, cv2.CV_64F)
        variances.append(float(np.var(lap)))

    if not variances:
        return 0.0

    mean_var = float(np.mean(variances))
    if mean_var < 1e-6:
        return 0.0

    # Coefficient of variation of Laplacian variances across strips
    cv = float(np.std(variances)) / mean_var

    # Map CV to score: CV of 0 = uniform sharpness (no depth), CV > 1 = strong depth
    # Sigmoid mapping: cv=0.5 -> ~0.5, cv=1.0 -> ~0.73
    score = 1.0 - np.exp(-cv * 1.5)
    return float(np.clip(score, 0.0, 1.0))


def _detect_lines(gray: np.ndarray) -> np.ndarray:
    """Detect line segments using LSD.

    Returns
    -------
    np.ndarray
        Array of shape (N, 4) with columns (x1, y1, x2, y2), or empty (0, 4).
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    result = lsd.detect(gray)
    lines_raw = result[0]
    if lines_raw is None or len(lines_raw) == 0:  # pyright: ignore[reportUnnecessaryComparison]
        return np.empty((0, 4), dtype=np.float32)

    # Shape: (N, 1, 4) -> (N, 4)
    lines: np.ndarray = lines_raw.reshape(-1, 4).astype(np.float32)

    # Filter by minimum length
    h, w = gray.shape[:2]
    diag = np.sqrt(float(h * h + w * w))
    min_length = diag * _MIN_LINE_LENGTH_FRAC

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx**2 + dy**2)
    mask = lengths >= min_length

    return lines[mask]


def _line_lengths_and_angles(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute lengths and angles (0-90 degrees) for line segments.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lengths, angles) where angles are in [0, 90] degrees.
    """
    if len(lines) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx**2 + dy**2)
    # Angle from horizontal, mapped to [0, 90]
    angles = np.abs(np.degrees(np.arctan2(np.abs(dy), np.abs(dx))))
    return lengths, angles


def measure_leading_lines(lines: np.ndarray, saliency: np.ndarray) -> float:
    """Measure convergence of detected lines toward the saliency centroid.

    Returns
    -------
    float
        Score in [0, 1]. Higher = lines converge toward visual interest.
    """
    h, w = saliency.shape[:2]
    if len(lines) == 0 or h < 3 or w < 3:
        return 0.0

    # Find saliency centroid
    s_sum = float(saliency.sum())
    if s_sum < 1e-6:
        # Default to center
        cx, cy = w / 2.0, h / 2.0
    else:
        com: tuple[float, float] = center_of_mass(saliency)  # pyright: ignore[reportAssignmentType]
        cy, cx = com[0], com[1]

    diag = np.sqrt(float(h * h + w * w))
    convergence_radius = diag * 0.15  # lines "converge" if they pass near centroid

    # For each line, compute distance from centroid to the line segment
    scores: list[float] = []
    lengths, _ = _line_lengths_and_angles(lines)

    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i]
        # Point-to-segment distance
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            continue
        t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = np.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2)

        # Score: 1.0 if line passes through centroid, decaying with distance
        line_score = max(0.0, 1.0 - dist / convergence_radius)
        scores.append(float(line_score * lengths[i]))

    if not scores:
        return 0.0

    total_length = float(lengths.sum())
    if total_length < 1e-6:
        return 0.0

    # Length-weighted convergence
    score = sum(scores) / total_length
    return float(np.clip(score, 0.0, 1.0))


def measure_diagonal_dominance(lines: np.ndarray) -> float:
    """Measure diagonal line energy from detected line segments.

    Diagonal lines (30-60 degrees from horizontal) create dynamism.

    Returns
    -------
    float
        Score in [0, 1]. Higher = stronger diagonal structure.
    """
    if len(lines) == 0:
        return 0.0

    lengths, angles = _line_lengths_and_angles(lines)
    total_energy = float(np.sum(lengths))
    if total_energy < 1e-6:
        return 0.0

    # Diagonal energy: lines in the 30-60 degree band
    diag_mask = (angles >= _DIAGONAL_LOW) & (angles <= _DIAGONAL_HIGH)
    diag_energy = float(np.sum(lengths[diag_mask]))

    ratio = diag_energy / total_energy

    # Boost: even 30% diagonal energy is significant
    score = min(1.0, ratio / 0.4)
    return float(np.clip(score, 0.0, 1.0))


def measure_negative_space(edges: np.ndarray) -> float:
    """Measure negative space via edge-density block thresholding.

    Low-density regions indicate open/empty areas. A balanced amount
    of negative space is compositionally desirable.

    Returns
    -------
    float
        Score in [0, 1]. Higher = well-distributed negative space.
    """
    h, w = edges.shape[:2]
    block_size = max(8, min(h, w) // 8)
    if block_size < 4:
        return 0.0

    # Compute block-wise edge density
    edge_float = (edges > 0).astype(np.float32)

    n_blocks_y = h // block_size
    n_blocks_x = w // block_size
    if n_blocks_y < 2 or n_blocks_x < 2:
        return 0.0

    densities: list[float] = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = edge_float[
                by * block_size : (by + 1) * block_size,
                bx * block_size : (bx + 1) * block_size,
            ]
            densities.append(float(block.mean()))

    densities_arr = np.array(densities)
    neg_space_ratio = float(np.mean(densities_arr < _NEGATIVE_SPACE_THRESHOLD))

    # Ideal negative space ratio: ~30-60% of blocks are "empty"
    # Score peaks at ~0.45 and drops toward 0 or 1
    if neg_space_ratio < 0.1:
        # Almost no negative space — cluttered
        score = neg_space_ratio / 0.1 * 0.3
    elif neg_space_ratio > 0.85:
        # Too much empty space — mostly blank
        score = (1.0 - neg_space_ratio) / 0.15 * 0.3
    else:
        # Good range: score based on being in the sweet spot
        # Peak at 0.45
        deviation = abs(neg_space_ratio - 0.45)
        score = max(0.3, 1.0 - deviation / 0.45)

    return float(np.clip(score, 0.0, 1.0))


def measure_framing(edges: np.ndarray) -> float:
    """Measure in-frame framing via border vs. center edge density ratio.

    High edge density in outer borders with lower density in center
    suggests in-frame elements bordering the subject.

    Returns
    -------
    float
        Score in [0, 1]. Higher = stronger framing effect.
    """
    h, w = edges.shape[:2]
    border_y = max(1, int(h * _FRAMING_BORDER_FRAC))
    border_x = max(1, int(w * _FRAMING_BORDER_FRAC))

    edge_float = (edges > 0).astype(np.float32)

    # Border region: outer strips
    top = edge_float[:border_y, :]
    bottom = edge_float[h - border_y :, :]
    left = edge_float[border_y : h - border_y, :border_x]
    right = edge_float[border_y : h - border_y, w - border_x :]

    border_pixels = np.concatenate(
        [top.ravel(), bottom.ravel(), left.ravel(), right.ravel()]
    )
    border_density = float(border_pixels.mean()) if len(border_pixels) > 0 else 0.0

    # Center region
    center = edge_float[border_y : h - border_y, border_x : w - border_x]
    center_density = float(center.mean()) if center.size > 0 else 0.0

    if center_density < 1e-6 and border_density < 1e-6:
        return 0.0

    # Framing = border has more edge activity than center
    ratio = 2.0 if center_density < 1e-6 else border_density / center_density

    # ratio > 1 suggests framing; ratio < 1 means center is busier
    if ratio <= 1.0:
        return 0.0

    # Map ratio to score: ratio of 1.5 -> ~0.5, ratio of 2.5+ -> ~1.0
    score = min(1.0, (ratio - 1.0) / 1.5)
    return float(np.clip(score, 0.0, 1.0))


# -- Tag Generation --


def _generate_tags(
    *,
    sub_scores: dict[str, float],
    saliency: np.ndarray,
    confidence_threshold: float,
) -> list[Tag]:
    """Generate composition tags from sub-property measurements."""
    tags: list[Tag] = []

    # Rule of thirds tag
    thirds = sub_scores["rule_of_thirds"]
    if thirds >= 0.6:
        tags.append(
            Tag(name="rule_of_thirds", confidence=thirds, category="composition")
        )

    # Centered composition detection (complementary to thirds)
    h, w = saliency.shape[:2]
    s_sum = float(saliency.sum())
    if s_sum > 1e-6 and h > 2 and w > 2:
        com: tuple[float, float] = center_of_mass(saliency)  # pyright: ignore[reportAssignmentType]
        cy, cx = h / 2.0, w / 2.0
        y_dev = abs(com[0] - cy) / cy
        x_dev = abs(com[1] - cx) / cx
        center_proximity = 1.0 - float(np.sqrt(y_dev**2 + x_dev**2) / np.sqrt(2.0))
        if center_proximity >= 0.85:
            tags.append(
                Tag(
                    name="centered",
                    confidence=center_proximity,
                    category="composition",
                )
            )

    # Balance tag
    balance = sub_scores["visual_balance"]
    if balance >= 0.7:
        tags.append(Tag(name="balanced", confidence=balance, category="composition"))

    # Symmetry tag
    symmetry = sub_scores["symmetry"]
    if symmetry >= 0.5:
        tags.append(Tag(name="symmetric", confidence=symmetry, category="composition"))

    # Leading lines tag
    leading = sub_scores["leading_lines"]
    if leading >= 0.4:
        tags.append(
            Tag(
                name="strong_leading_lines",
                confidence=leading,
                category="composition",
            )
        )

    # Diagonal dominance tag
    diagonal = sub_scores["diagonal_dominance"]
    if diagonal >= 0.4:
        tags.append(
            Tag(
                name="diagonal_composition",
                confidence=diagonal,
                category="composition",
            )
        )

    # Negative space tag
    neg_space = sub_scores["negative_space"]
    if neg_space >= 0.7:
        tags.append(
            Tag(
                name="open_composition",
                confidence=neg_space,
                category="composition",
            )
        )

    # Framing tag
    framing = sub_scores["framing"]
    if framing >= 0.4:
        tags.append(
            Tag(
                name="framed_subject",
                confidence=framing,
                category="composition",
            )
        )

    return [t for t in tags if t.confidence >= confidence_threshold]


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


# -- Analyzer Class --


class CompositionAnalyzer:
    """Composition dimension analyzer — fully classical, no model dependencies.

    Measures eight sub-properties of spatial arrangement quality and produces
    a combined score with descriptive tags.
    """

    name: str = "composition"

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        """Analyze composition properties of an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array with shape (H, W, 3).
        config : AnalyzerConfig
            Per-analyzer configuration.
        shared : SharedModels
            Outputs from shared model inference (unused in Phase 2).

        Returns
        -------
        AnalyzerResult
            Composition dimension score, tags, and sub-property metadata.
        """
        # Prepare grayscale at analysis resolution
        gray, _scale = _prepare_gray(image)
        h, w = gray.shape[:2]

        # Edge detection (shared across sub-properties)
        edges: np.ndarray = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)

        # Saliency map
        saliency = compute_saliency(gray)

        # Line detection (shared by leading_lines and diagonal_dominance)
        lines = _detect_lines(gray)

        # Compute all sub-properties
        sub_scores: dict[str, float] = {
            "rule_of_thirds": measure_rule_of_thirds(saliency),
            "visual_balance": measure_visual_balance(saliency),
            "symmetry": measure_symmetry(edges),
            "depth_layering": measure_depth_layering(gray),
            "leading_lines": measure_leading_lines(lines, saliency),
            "diagonal_dominance": measure_diagonal_dominance(lines),
            "negative_space": measure_negative_space(edges),
            "framing": measure_framing(edges),
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
            sub_scores=sub_scores,
            saliency=saliency,
            confidence_threshold=config.confidence_threshold,
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "sub_scores": sub_scores,
            "weights_used": weights,
            "n_lines_detected": len(lines),
            "analysis_resolution": [h, w],
        }

        return AnalyzerResult(
            analyzer=self.name,
            score=score,
            tags=tags,
            metadata=metadata,
        )
