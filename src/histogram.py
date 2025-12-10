import numpy as np

from .utils import ensure_grayscale, normalize_image


def compute_histogram(img: np.ndarray) -> np.ndarray:
    """Compute histogram (256 bins) for grayscale image."""
    gray = ensure_grayscale(img)
    hist = np.zeros(256, dtype=np.int32)
    flat = gray.flatten()
    for val in flat:
        idx = int(np.clip(val, 0, 255))
        hist[idx] += 1
    return hist


def histogram_evaluation(hist: np.ndarray) -> str:
    """Provide simple qualitative assessment."""
    total = np.sum(hist)
    if total == 0:
        return "Empty histogram."
    spread = np.count_nonzero(hist)
    low = np.sum(hist[:32]) / total
    high = np.sum(hist[-32:]) / total
    if spread > 220 and low < 0.15 and high < 0.15:
        return "Histogram is well-spread with good contrast."
    if spread < 80:
        return "Histogram is narrow: image likely low-contrast."
    if low > 0.3:
        return "Too many dark pixels: image underexposed."
    if high > 0.3:
        return "Too many bright pixels: image overexposed."
    return "Histogram is balanced but could be enhanced."


def equalize(img: np.ndarray) -> np.ndarray:
    """Manual histogram equalization for grayscale image."""
    gray = ensure_grayscale(img)
    hist = compute_histogram(gray)
    cdf = np.cumsum(hist).astype(np.float32)
    cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
    lut = (cdf_norm * 255).astype(np.uint8)
    flat = gray.flatten().astype(np.uint8)
    equalized = lut[flat].reshape(gray.shape)
    return equalized.astype(np.float32)


