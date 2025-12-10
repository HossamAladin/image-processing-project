import io
import math
from typing import Tuple

import numpy as np
from PIL import Image


def load_image_to_array(path: str) -> np.ndarray:
    """Load an image file into a numpy array (RGB)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a numpy array back to a PIL image with proper clipping."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def ensure_grayscale(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is grayscale."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Use luminance weights to convert.
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError("Unsupported image shape for grayscale conversion")


def normalize_image(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0..255 for visualization."""
    arr = arr.astype(np.float32)
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v < 1e-5:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - min_v) * 255.0 / (max_v - min_v)
    return norm.astype(np.uint8)


def image_info(arr: np.ndarray, file_size: int = 0, mode_hint: str = "") -> dict:
    """Return dictionary with metadata."""
    h, w = arr.shape[:2]
    mode = mode_hint or ("Grayscale" if arr.ndim == 2 else "RGB")
    return {
        "width": w,
        "height": h,
        "size_bytes": file_size,
        "mode": mode,
    }


def bilinear_interpolate(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample image at fractional coordinates using bilinear interpolation."""
    h, w = img.shape[:2]
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return np.zeros(img.shape[2:], dtype=np.float32) if img.ndim == 3 else 0.0
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x0, y - y0
    if img.ndim == 2:
        val = (
            (1 - dx) * (1 - dy) * img[y0, x0]
            + dx * (1 - dy) * img[y0, x0 + 1]
            + (1 - dx) * dy * img[y0 + 1, x0]
            + dx * dy * img[y0 + 1, x0 + 1]
        )
        return val
    # Color image
    top = (1 - dx) * img[y0, x0] + dx * img[y0, x0 + 1]
    bottom = (1 - dx) * img[y0 + 1, x0] + dx * img[y0 + 1, x0 + 1]
    return (1 - dy) * top + dy * bottom


def pad_image(img: np.ndarray, pad_y: int, pad_x: int, mode: str = "edge") -> np.ndarray:
    """Pad an image for convolution."""
    return np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)) if img.ndim == 3 else ((pad_y, pad_y), (pad_x, pad_x)), mode=mode)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generate a Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def convolve_image(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual convolution for grayscale or RGB images."""
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = pad_image(img, pad_y, pad_x)
    h, w = img.shape[:2]
    if img.ndim == 2:
        out = np.zeros_like(img, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                region = padded[y : y + kh, x : x + kw]
                out[y, x] = np.sum(region * kernel)
        return out
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            region = padded[y : y + kh, x : x + kw, :]
            out[y, x] = np.sum(region * kernel[:, :, None], axis=(0, 1))
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio."""
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(m))


def compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """Compute compression ratio (original / compressed)."""
    if compressed_bits == 0:
        return 0.0
    return original_bits / compressed_bits


