import math
from typing import Callable

import numpy as np

from .utils import bilinear_interpolate


def _apply_affine(img: np.ndarray, matrix: np.ndarray, output_shape=None) -> np.ndarray:
    """Apply affine matrix to image using backward mapping and bilinear interpolation."""
    h, w = img.shape[:2]
    if output_shape is None:
        output_shape = (h, w)
    out_h, out_w = output_shape
    out = np.zeros((out_h, out_w, img.shape[2]) if img.ndim == 3 else (out_h, out_w), dtype=np.float32)

    inv_m = np.linalg.inv(matrix)
    for y_out in range(out_h):
        for x_out in range(out_w):
            src = inv_m @ np.array([x_out, y_out, 1.0])
            x_src, y_src = src[0], src[1]
            if 0 <= x_src < w - 1 and 0 <= y_src < h - 1:
                out[y_out, x_out] = bilinear_interpolate(img, x_src, y_src)
    return out


def translate(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    m = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    return _apply_affine(img, m)


def scale(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
    out_shape = (int(img.shape[0] * sy), int(img.shape[1] * sx))
    return _apply_affine(img, m, output_shape=out_shape)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    m = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    return _apply_affine(img, m)


def shear_x(img: np.ndarray, shx: float) -> np.ndarray:
    m = np.array([[1, shx, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    out_shape = (img.shape[0], int(img.shape[1] + abs(shx) * img.shape[0]))
    return _apply_affine(img, m, output_shape=out_shape)


def shear_y(img: np.ndarray, shy: float) -> np.ndarray:
    m = np.array([[1, 0, 0], [shy, 1, 0], [0, 0, 1]], dtype=np.float32)
    out_shape = (int(img.shape[0] + abs(shy) * img.shape[1]), img.shape[1])
    return _apply_affine(img, m, output_shape=out_shape)


