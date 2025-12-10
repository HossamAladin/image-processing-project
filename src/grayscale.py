import numpy as np


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Manual grayscale conversion using luminance weights."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


