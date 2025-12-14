import numpy as np


def compress(img: np.ndarray):
    """Simple DPCM using left neighbor prediction."""
    gray = img.astype(np.float32)
    h, w = gray.shape
    diff = np.zeros_like(gray)
    for y in range(h):
        for x in range(w):
            pred = gray[y, x - 1] if x > 0 else 0
            diff[y, x] = gray[y, x] - pred
    return diff, gray.shape


def decompress(diff: np.ndarray, shape):
    h, w = shape
    recon = np.zeros(shape, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            pred = recon[y, x - 1] if x > 0 else 0
            recon[y, x] = diff[y, x] + pred
    return recon


