import numpy as np

from .utils import convolve_image, ensure_grayscale, normalize_image


def laplacian(img: np.ndarray) -> np.ndarray:
    gray = ensure_grayscale(img)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    out = convolve_image(gray, kernel)
    return normalize_image(out)


def sobel(img: np.ndarray) -> np.ndarray:
    gray = ensure_grayscale(img)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve_image(gray, kx)
    gy = convolve_image(gray, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return normalize_image(mag)


def gradient(img: np.ndarray) -> np.ndarray:
    """First-derivative gradient filter (simple forward difference)."""
    gray = ensure_grayscale(img)
    kx = np.array([[0, -1, 1]], dtype=np.float32)
    ky = np.array([[0], [-1], [1]], dtype=np.float32)
    gx = convolve_image(gray, kx)
    gy = convolve_image(gray, ky)
    mag = np.abs(gx) + np.abs(gy)
    return normalize_image(mag)


