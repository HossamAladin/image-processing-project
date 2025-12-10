import numpy as np


def threshold_average(gray: np.ndarray) -> float:
    """Compute average intensity as threshold."""
    if gray.ndim != 2:
        raise ValueError("Binary conversion expects grayscale image")
    return float(np.mean(gray))


def to_binary(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale to binary using average threshold."""
    t = threshold_average(gray)
    binary = (gray >= t).astype(np.uint8) * 255
    return binary


def threshold_evaluation(gray: np.ndarray, threshold: float) -> str:
    """Provide a simple evaluation of threshold quality."""
    low_ratio = np.mean(gray < threshold)
    high_ratio = 1.0 - low_ratio
    if 0.45 < low_ratio < 0.55:
        return "Threshold is well-balanced between dark and bright regions."
    if high_ratio < 0.2:
        return "Threshold too high: image may become overly dark."
    if low_ratio < 0.2:
        return "Threshold too low: image may become overly bright."
    return "Threshold is acceptable but could be adjusted for better separation."


