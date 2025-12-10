import numpy as np

from .utils import convolve_image, gaussian_kernel, pad_image


def gaussian_blur(img: np.ndarray, size: int = 19, sigma: float = 3.0) -> np.ndarray:
    """Manual Gaussian blur via convolution."""
    kernel = gaussian_kernel(size, sigma)
    blurred = convolve_image(img, kernel)
    return blurred


def median_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    """Manual median filter."""
    pad = size // 2
    padded = pad_image(img, pad, pad)
    h, w = img.shape[:2]
    if img.ndim == 2:
        out = np.zeros_like(img, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                region = padded[y : y + size, x : x + size]
                out[y, x] = np.median(region)
        return out
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            region = padded[y : y + size, x : x + size, :]
            out[y, x, 0] = np.median(region[:, :, 0])
            out[y, x, 1] = np.median(region[:, :, 1])
            out[y, x, 2] = np.median(region[:, :, 2])
    return out


