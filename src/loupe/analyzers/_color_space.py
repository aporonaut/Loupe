"""OkLab/OkLCh color space conversion utilities.

Provides sRGB → linear RGB → OkLab → OkLCh conversions operating on
(H, W, 3) float32 arrays. Implementation follows Björn Ottosson's OkLab
specification using two 3x3 matrix multiplications with cube-root nonlinearity.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Gamma-decode sRGB [0, 1] to linear RGB [0, 1].

    Parameters
    ----------
    srgb : np.ndarray
        sRGB values in [0, 1], any shape with last dim = 3.

    Returns
    -------
    np.ndarray
        Linear RGB values, float32, same shape.
    """
    srgb = srgb.astype(np.float32)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_oklab(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB to OkLab.

    Parameters
    ----------
    linear_rgb : np.ndarray
        Linear sRGB values in [0, 1], shape (..., 3).

    Returns
    -------
    np.ndarray
        OkLab (L, a, b) values, float32, same shape.
    """
    # Linear RGB → LMS (Ottosson's M1 matrix)
    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        dtype=np.float32,
    )
    # LMS^(1/3) → OkLab (Ottosson's M2 matrix)
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        dtype=np.float32,
    )
    lms = linear_rgb.astype(np.float32) @ m1.T
    # Cube root, handling negative values from floating point noise
    lms_g = np.sign(lms) * np.abs(lms) ** (1.0 / 3.0)
    return lms_g @ m2.T


def oklab_to_oklch(lab: np.ndarray) -> np.ndarray:
    """Convert OkLab to OkLCh (cylindrical).

    Parameters
    ----------
    lab : np.ndarray
        OkLab (L, a, b) values, shape (..., 3).

    Returns
    -------
    np.ndarray
        OkLCh (L, C, h) where h is in degrees [0, 360), float32, same shape.
    """
    l_vals = lab[..., 0]
    a_vals = lab[..., 1]
    b_vals = lab[..., 2]
    c_vals = np.sqrt(a_vals**2 + b_vals**2).astype(np.float32)
    h_vals = np.degrees(np.arctan2(b_vals, a_vals)).astype(np.float32) % 360.0
    return np.stack([l_vals, c_vals, h_vals], axis=-1).astype(np.float32)


def srgb_uint8_to_oklab(image: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 image to OkLab in one step.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 array, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        OkLab (L, a, b) values, float32, shape (H, W, 3).
    """
    return linear_to_oklab(srgb_to_linear(image.astype(np.float32) / 255.0))


def srgb_uint8_to_oklch(image: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 image to OkLCh in one step.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 array, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        OkLCh (L, C, h) values, float32, shape (H, W, 3).
    """
    return oklab_to_oklch(srgb_uint8_to_oklab(image))
