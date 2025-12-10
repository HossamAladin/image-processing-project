import numpy as np

from .utils import bilinear_interpolate


def nearest_neighbor(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Nearest neighbor interpolation."""
    h, w = img.shape[:2]
    out = np.zeros((new_h, new_w, img.shape[2]) if img.ndim == 3 else (new_h, new_w), dtype=np.float32)
    for y in range(new_h):
        for x in range(new_w):
            src_x = int(round(x * (w - 1) / max(new_w - 1, 1)))
            src_y = int(round(y * (h - 1) / max(new_h - 1, 1)))
            out[y, x] = img[src_y, src_x]
    return out


def bilinear(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Bilinear interpolation."""
    h, w = img.shape[:2]
    out = np.zeros((new_h, new_w, img.shape[2]) if img.ndim == 3 else (new_h, new_w), dtype=np.float32)
    for y in range(new_h):
        for x in range(new_w):
            src_x = x * (w - 1) / max(new_w - 1, 1)
            src_y = y * (h - 1) / max(new_h - 1, 1)
            out[y, x] = bilinear_interpolate(img, src_x, src_y)
    return out


def _cubic_weight(t: float) -> float:
    """Cubic weight (Catmull-Rom spline)."""
    a = -0.5
    t = abs(t)
    if t <= 1:
        return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
    if t < 2:
        return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
    return 0.0


def bicubic(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Bicubic interpolation using Catmull-Rom kernel."""
    h, w = img.shape[:2]
    out = np.zeros((new_h, new_w, img.shape[2]) if img.ndim == 3 else (new_h, new_w), dtype=np.float32)
    for y in range(new_h):
        for x in range(new_w):
            src_x = x * (w - 1) / max(new_w - 1, 1)
            src_y = y * (h - 1) / max(new_h - 1, 1)
            x0, y0 = int(np.floor(src_x)), int(np.floor(src_y))
            dx, dy = src_x - x0, src_y - y0
            accum = np.zeros(img.shape[2]) if img.ndim == 3 else 0.0
            total_weight = 0.0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    xm = min(max(x0 + n, 0), w - 1)
                    ym = min(max(y0 + m, 0), h - 1)
                    weight = _cubic_weight(n - dx) * _cubic_weight(m - dy)
                    accum += img[ym, xm] * weight
                    total_weight += weight
            out[y, x] = accum / (total_weight + 1e-8)
    return out


