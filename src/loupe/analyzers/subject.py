# Copyright 2025 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Subject analyzer — focal emphasis analysis.

Measures how effectively a frame emphasizes its subject(s) across six
sub-properties: saliency strength, figure-ground separation, DOF effect,
negative space utilization, subject completeness, and subject scale.

Learned models identify the subject (anime segmentation mask + detection
boxes from SharedModels); classical CV measures how well the frame
emphasizes it.

Tags produced:
    extreme_closeup — subject occupies >60% of frame area
    closeup — subject occupies 30-60% of frame area
    medium_shot — subject occupies 15-30% of frame area
    wide_shot — subject occupies 5-15% of frame area
    very_wide — subject occupies <5% of frame area
    environment_focus — no character detected (landscape/scenery)
    strong_separation — strong figure-ground separation
    shallow_dof — significant DOF blur differential
    complete_subject — subject fully within frame (no cropping)

Known limitations:
    - Requires segmentation mask from SharedModels. Without it, the
      analyzer returns a low score and the ``environment_focus`` tag.
    - Saliency uses OpenCV's spectral residual method, which has moderate
      reliability on anime content.
    - Multi-subject compositions use the union of all character masks.
    - Calibration is initial/theoretical — will be refined with real images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from loupe.analyzers._color_space import srgb_uint8_to_oklab
from loupe.core.models import AnalyzerResult, Tag

if TYPE_CHECKING:
    from loupe.analyzers.base import AnalyzerConfig, DetectionBox, SharedModels

# -- Constants --

# Default sub-property weights from RQ2 §8.6
DEFAULT_WEIGHTS: dict[str, float] = {
    "saliency_strength": 0.15,
    "figure_ground_separation": 0.25,
    "dof_effect": 0.15,
    "negative_space_utilization": 0.15,
    "subject_completeness": 0.15,
    "subject_scale": 0.15,
}

# Segmentation mask binarization threshold
_MASK_THRESHOLD = 0.5

# Boundary dilation kernel size for figure-ground measurement
_BOUNDARY_DILATION_PX = 5

# Edge density block size for negative space computation
_NEG_SPACE_BLOCK_FRAC = 1 / 8

# Edge density threshold for "quiet" blocks
_NEG_SPACE_EDGE_THRESHOLD = 0.05

# Scale categorization thresholds (fraction of frame area)
_SCALE_EXTREME_CLOSEUP = 0.60
_SCALE_CLOSEUP = 0.30
_SCALE_MEDIUM = 0.15
_SCALE_WIDE = 0.05


# -- Mask Utilities --


def _get_subject_mask(shared: SharedModels) -> np.ndarray | None:
    """Extract binary subject mask from shared models.

    Parameters
    ----------
    shared : SharedModels
        Shared model outputs (may or may not contain segmentation_mask).

    Returns
    -------
    np.ndarray | None
        Binary mask (H, W) uint8 with 1=subject, 0=background.
        None if no segmentation mask is available.
    """
    seg_mask = shared.get("segmentation_mask")
    if seg_mask is None:
        return None
    return (seg_mask > _MASK_THRESHOLD).astype(np.uint8)


def _subject_area_ratio(mask: np.ndarray) -> float:
    """Compute fraction of frame occupied by subject mask."""
    total = mask.size
    if total == 0:
        return 0.0
    return float(np.sum(mask > 0)) / total


# -- Sub-Property Measurements --


def _spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
    """Compute spectral residual saliency map (Hou & Zhang 2007).

    Manual implementation — avoids dependency on opencv-contrib's
    ``cv2.saliency`` module which is not in opencv-python-headless.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image (H, W).

    Returns
    -------
    np.ndarray
        Saliency map (H, W) float32 in [0, 1].
    """
    # Resize to small fixed size for FFT efficiency (original paper uses 64x64)
    small: np.ndarray = cv2.resize(gray, (64, 64)).astype(np.float32)

    # Compute log-amplitude spectrum
    f = np.fft.fft2(small)
    amplitude = np.abs(f)
    phase = np.angle(f)

    log_amplitude = np.log(amplitude + 1e-10)

    # Spectral residual: subtract average (smoothed) log-amplitude
    avg_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    smoothed: np.ndarray = cv2.filter2D(
        log_amplitude.astype(np.float32), -1, avg_kernel
    )
    residual = log_amplitude - smoothed

    # Reconstruct saliency via inverse FFT of residual + original phase
    saliency_complex = np.exp(residual + 1j * phase)
    saliency_small = np.abs(np.fft.ifft2(saliency_complex)) ** 2

    # Gaussian blur to smooth the saliency map
    saliency_small_f32 = saliency_small.astype(np.float32)
    saliency_blurred: np.ndarray = cv2.GaussianBlur(saliency_small_f32, (7, 7), 2.5)

    # Resize back to original dimensions
    h, w = gray.shape[:2]
    sal_map: np.ndarray = cv2.resize(saliency_blurred, (w, h))

    # Normalize to [0, 1]
    sal_max = sal_map.max()
    if sal_max > 0:
        sal_map = sal_map / sal_max

    return sal_map


def measure_saliency_strength(image: np.ndarray, mask: np.ndarray) -> float:
    """Measure concentration of spectral residual saliency within subject mask.

    Computes a spectral residual saliency map (Hou & Zhang 2007), then
    measures what fraction of saliency mass falls within the subject region.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    mask : np.ndarray
        Binary subject mask (H, W) uint8.

    Returns
    -------
    float
        Score in [0, 1]. Higher = saliency concentrated on subject.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sal_map = _spectral_residual_saliency(gray)

    total_saliency = float(np.sum(sal_map))
    if total_saliency < 1e-6:
        return 0.0

    subject_saliency = float(np.sum(sal_map * mask.astype(np.float32)))
    concentration = subject_saliency / total_saliency

    # Adjust for subject area: a large subject capturing most saliency
    # is less impressive than a small subject doing the same.
    area_ratio = _subject_area_ratio(mask)
    if area_ratio > 0.01:
        # Expected concentration if saliency were uniform = area_ratio
        # Relative concentration: how much above chance
        relative = concentration / area_ratio
        # Map: relative=1 (chance) → ~0.2, relative=3+ → ~0.9
        score = 1.0 - np.exp(-relative * 0.5)
    else:
        score = concentration

    return float(np.clip(score, 0.0, 1.0))


def measure_figure_ground_separation(
    image: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Measure figure-ground separation via Lab Delta-E and boundary edge strength.

    Computes the perceptual color difference between foreground and background
    regions (in OkLab), plus the Sobel gradient magnitude at the mask boundary.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    mask : np.ndarray
        Binary subject mask (H, W) uint8.

    Returns
    -------
    float
        Score in [0, 1]. Higher = stronger figure-ground separation.
    """
    fg_mask = mask > 0
    bg_mask = ~fg_mask

    fg_count = int(np.sum(fg_mask))
    bg_count = int(np.sum(bg_mask))

    if fg_count == 0 or bg_count == 0:
        return 0.0

    # OkLab color difference between FG and BG mean colors
    oklab = srgb_uint8_to_oklab(image)
    fg_pixels = oklab[fg_mask]
    bg_pixels = oklab[bg_mask]
    fg_mean = np.mean(fg_pixels, axis=0)
    bg_mean = np.mean(bg_pixels, axis=0)
    delta_e = float(np.linalg.norm(fg_mean - bg_mean))

    # Normalize: OkLab Delta-E of ~0.3 is a strong difference
    color_score = min(1.0, delta_e / 0.3)

    # Boundary edge strength via Sobel at mask boundary
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag: np.ndarray = np.sqrt(sobel_x**2 + sobel_y**2)  # pyright: ignore[reportUnknownArgumentType]

    # Extract boundary pixels (dilate - erode of mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_BOUNDARY_DILATION_PX, _BOUNDARY_DILATION_PX)
    )
    dilated: np.ndarray = cv2.dilate(mask, kernel, iterations=1)
    eroded: np.ndarray = cv2.erode(mask, kernel, iterations=1)
    boundary = (dilated - eroded) > 0

    boundary_count = int(np.sum(boundary))
    if boundary_count > 0:
        boundary_gradient = float(np.mean(gradient_mag[boundary]))
        # Normalize: gradient magnitude of ~50+ on uint8 is strong
        edge_score = min(1.0, boundary_gradient / 50.0)
    else:
        edge_score = 0.0

    # Combined: color difference matters more
    return float(np.clip(0.6 * color_score + 0.4 * edge_score, 0.0, 1.0))


def measure_dof_effect(image: np.ndarray, mask: np.ndarray) -> float:
    """Measure depth-of-field effect via Laplacian variance ratio.

    Compares sharpness (Laplacian variance) of subject vs background regions.
    Higher ratio = subject is sharper than background (DOF blur present).

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    mask : np.ndarray
        Binary subject mask (H, W) uint8.

    Returns
    -------
    float
        Score in [0, 1]. Higher = stronger DOF effect.
    """
    fg_mask = mask > 0
    bg_mask = ~fg_mask

    fg_count = int(np.sum(fg_mask))
    bg_count = int(np.sum(bg_mask))

    if fg_count < 10 or bg_count < 10:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    fg_var = float(np.var(laplacian[fg_mask]))
    bg_var = float(np.var(laplacian[bg_mask]))

    if bg_var < 1e-6:
        # Background has zero variance — completely flat (synthetic edge case)
        return 0.5 if fg_var > 1e-6 else 0.0

    ratio = fg_var / bg_var

    # DOF effect: subject sharper than background → ratio > 1
    # ratio=1 → no DOF, ratio=3+ → strong DOF
    # ratio <= 1 means subject blurrier than background (unusual)
    score = 0.1 if ratio <= 1.0 else 1.0 - np.exp(-(ratio - 1.0) * 0.5)

    return float(np.clip(score, 0.0, 1.0))


def measure_negative_space_utilization(
    image: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Measure quiet/low-complexity area distribution around the subject.

    Uses edge density thresholding to find calm regions, then measures
    their spatial distribution relative to the subject.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    mask : np.ndarray
        Binary subject mask (H, W) uint8.

    Returns
    -------
    float
        Score in [0, 1]. Higher = well-distributed quiet space around subject.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_float = (edges > 0).astype(np.float32)

    block_size = max(8, int(min(h, w) * _NEG_SPACE_BLOCK_FRAC))
    n_blocks_y = h // block_size
    n_blocks_x = w // block_size

    if n_blocks_y < 2 or n_blocks_x < 2:
        return 0.0

    # Classify blocks as quiet or busy, and as subject or background
    bg_quiet_count = 0
    bg_total_count = 0

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * block_size
            y1 = y0 + block_size
            x0 = bx * block_size
            x1 = x0 + block_size

            block_mask = mask[y0:y1, x0:x1]
            # Block is background if <50% subject
            if float(np.mean(block_mask)) < 0.5:
                bg_total_count += 1
                block_density = float(edge_float[y0:y1, x0:x1].mean())
                if block_density < _NEG_SPACE_EDGE_THRESHOLD:
                    bg_quiet_count += 1

    if bg_total_count == 0:
        return 0.0

    quiet_ratio = bg_quiet_count / bg_total_count

    # A moderate amount of quiet background is ideal for negative space
    # Sweet spot: 40-70% of background blocks are quiet
    if quiet_ratio < 0.1:
        score = quiet_ratio * 3.0  # Very busy background
    elif quiet_ratio > 0.9:
        score = 0.5 + 0.5 * (1.0 - quiet_ratio) / 0.1  # Almost entirely empty
    else:
        # Good range — peak near 0.55
        deviation = abs(quiet_ratio - 0.55)
        score = max(0.4, 1.0 - deviation / 0.45)

    return float(np.clip(score, 0.0, 1.0))


def measure_subject_completeness(mask: np.ndarray) -> float:
    """Measure subject completeness by checking mask contact with frame edges.

    Low edge contact = complete subject (not cropped by frame boundary).
    High edge contact = subject extends beyond frame (cropped).

    Parameters
    ----------
    mask : np.ndarray
        Binary subject mask (H, W) uint8.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more complete (less cropped) subject.
    """
    h, w = mask.shape[:2]
    if h < 2 or w < 2:
        return 0.0

    subject_pixels = int(np.sum(mask > 0))
    if subject_pixels == 0:
        return 0.0

    # Count subject pixels touching each frame edge
    top_contact = int(np.sum(mask[0, :] > 0))
    bottom_contact = int(np.sum(mask[h - 1, :] > 0))
    left_contact = int(np.sum(mask[:, 0] > 0))
    right_contact = int(np.sum(mask[:, w - 1] > 0))

    total_edge_contact = top_contact + bottom_contact + left_contact + right_contact
    total_edge_length = 2 * (h + w)

    edge_contact_ratio = total_edge_contact / total_edge_length

    # Low contact → complete subject. Score decreases with contact.
    # Mild contact on one edge (e.g., feet at bottom) is acceptable.
    # edge_contact_ratio=0 → 1.0, =0.1 → ~0.6, =0.3+ → ~0.2
    score = np.exp(-edge_contact_ratio * 10.0)
    return float(np.clip(score, 0.0, 1.0))


def measure_subject_scale(
    mask: np.ndarray,
    detection_boxes: list[DetectionBox] | None = None,
) -> float:
    """Measure subject scale (area ratio) and score informativeness.

    Moderate scales (medium shot, close-up) are generally more informative
    for wallpaper use than extreme close-ups or very wide shots.

    Parameters
    ----------
    mask : np.ndarray
        Binary subject mask (H, W) uint8.
    detection_boxes : list[DetectionBox] | None
        Optional detection boxes for additional scale context.

    Returns
    -------
    float
        Score in [0, 1]. Higher = more informative scale for wallpaper use.
    """
    area_ratio = _subject_area_ratio(mask)

    if area_ratio < 0.01:
        # Almost no subject — very wide or no subject
        return 0.1

    # Preferred scale range: 15-50% (medium to close-up)
    # Score peaks in the sweet spot and falls off at extremes
    if area_ratio < _SCALE_WIDE:
        # Very wide: subject too small for emphasis
        score = 0.2 + 0.3 * (area_ratio / _SCALE_WIDE)
    elif area_ratio < _SCALE_MEDIUM:
        # Wide shot: decent, increasing toward medium
        score = 0.5 + 0.3 * ((area_ratio - _SCALE_WIDE) / (_SCALE_MEDIUM - _SCALE_WIDE))
    elif area_ratio < _SCALE_CLOSEUP:
        # Medium shot: ideal range
        score = 0.8 + 0.2 * (
            (area_ratio - _SCALE_MEDIUM) / (_SCALE_CLOSEUP - _SCALE_MEDIUM)
        )
    elif area_ratio < _SCALE_EXTREME_CLOSEUP:
        # Close-up: still good, starts to decline
        progress = (area_ratio - _SCALE_CLOSEUP) / (
            _SCALE_EXTREME_CLOSEUP - _SCALE_CLOSEUP
        )
        score = 1.0 - 0.2 * progress
    else:
        # Extreme close-up: subject dominates, less compositionally interesting
        score = 0.6 - 0.3 * min(1.0, (area_ratio - _SCALE_EXTREME_CLOSEUP) / 0.3)

    return float(np.clip(score, 0.0, 1.0))


# -- Scale Categorization --


def _categorize_scale(area_ratio: float) -> str:
    """Categorize subject scale from area ratio."""
    if area_ratio >= _SCALE_EXTREME_CLOSEUP:
        return "extreme_closeup"
    if area_ratio >= _SCALE_CLOSEUP:
        return "closeup"
    if area_ratio >= _SCALE_MEDIUM:
        return "medium_shot"
    if area_ratio >= _SCALE_WIDE:
        return "wide_shot"
    return "very_wide"


# -- Tag Generation --


def _generate_tags(
    *,
    area_ratio: float,
    sub_scores: dict[str, float],
    has_subject: bool,
    confidence_threshold: float,
) -> list[Tag]:
    """Generate subject tags from analysis results."""
    tags: list[Tag] = []

    if not has_subject:
        tags.append(Tag(name="environment_focus", confidence=0.9, category="subject"))
        return tags

    # Scale tag (always generated when subject exists)
    scale_name = _categorize_scale(area_ratio)
    # Confidence based on how clearly the scale fits a category
    tags.append(Tag(name=scale_name, confidence=0.8, category="subject"))

    # Strong figure-ground separation
    fg_score = sub_scores.get("figure_ground_separation", 0.0)
    if fg_score >= 0.7:
        tags.append(
            Tag(
                name="strong_separation",
                confidence=fg_score,
                category="subject",
            )
        )

    # Shallow DOF
    dof_score = sub_scores.get("dof_effect", 0.0)
    if dof_score >= 0.5:
        tags.append(Tag(name="shallow_dof", confidence=dof_score, category="subject"))

    # Complete subject (not cropped)
    completeness = sub_scores.get("subject_completeness", 0.0)
    if completeness >= 0.7:
        tags.append(
            Tag(
                name="complete_subject",
                confidence=completeness,
                category="subject",
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


class SubjectAnalyzer:
    """Subject dimension analyzer — measures focal emphasis quality.

    Uses shared segmentation mask and detection boxes from the engine's
    model infrastructure to identify the subject, then applies classical
    CV measurements to assess how well the frame emphasizes it.
    """

    name: str = "subject"

    def analyze(
        self,
        image: np.ndarray,
        config: AnalyzerConfig,
        shared: SharedModels,
    ) -> AnalyzerResult:
        """Analyze subject emphasis properties of an image.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array with shape (H, W, 3).
        config : AnalyzerConfig
            Per-analyzer configuration.
        shared : SharedModels
            Outputs from shared model inference (requires segmentation_mask).

        Returns
        -------
        AnalyzerResult
            Subject dimension score, tags, and sub-property metadata.
        """
        mask = _get_subject_mask(shared)
        detection_boxes = shared.get("detection_boxes")

        # No segmentation mask → environment focus with low score
        if mask is None or int(np.sum(mask > 0)) == 0:
            tags = _generate_tags(
                area_ratio=0.0,
                sub_scores={},
                has_subject=False,
                confidence_threshold=config.confidence_threshold,
            )
            return AnalyzerResult(
                analyzer=self.name,
                score=0.1,
                tags=tags,
                metadata={
                    "has_subject": False,
                    "subject_area_ratio": 0.0,
                    "reason": "no_segmentation_mask" if mask is None else "empty_mask",
                },
            )

        # Resize mask to match image dimensions if needed
        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        area_ratio = _subject_area_ratio(mask)

        # Compute all sub-properties
        sub_scores: dict[str, float] = {
            "saliency_strength": measure_saliency_strength(image, mask),
            "figure_ground_separation": measure_figure_ground_separation(image, mask),
            "dof_effect": measure_dof_effect(image, mask),
            "negative_space_utilization": measure_negative_space_utilization(
                image, mask
            ),
            "subject_completeness": measure_subject_completeness(mask),
            "subject_scale": measure_subject_scale(mask, detection_boxes),
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
            area_ratio=area_ratio,
            sub_scores=sub_scores,
            has_subject=True,
            confidence_threshold=config.confidence_threshold,
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "has_subject": True,
            "subject_area_ratio": round(area_ratio, 4),
            "scale_category": _categorize_scale(area_ratio),
            "sub_scores": sub_scores,
            "weights_used": weights,
        }

        return AnalyzerResult(
            analyzer=self.name,
            score=score,
            tags=tags,
            metadata=metadata,
        )
