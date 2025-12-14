import numpy as np


def haar_forward(gray: np.ndarray):
    gray = gray.astype(np.float32)
    h, w = gray.shape
    # Pad to even dimensions if needed
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h > 0 or pad_w > 0:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
        h, w = gray.shape
    out = gray.copy()
    # Row transform
    temp = np.zeros_like(out)
    for y in range(h):
        for x in range(0, w - 1, 2):
            a = (out[y, x] + out[y, x + 1]) / 2
            d = (out[y, x] - out[y, x + 1]) / 2
            temp[y, x // 2] = a
            temp[y, x // 2 + w // 2] = d
        # Handle odd width
        if w % 2 == 1:
            temp[y, w // 2] = out[y, w - 1]
    out = temp.copy()
    # Column transform
    temp = np.zeros_like(out)
    for x in range(w):
        for y in range(0, h - 1, 2):
            a = (out[y, x] + out[y + 1, x]) / 2
            d = (out[y, x] - out[y + 1, x]) / 2
            temp[y // 2, x] = a
            temp[y // 2 + h // 2, x] = d
        # Handle odd height
        if h % 2 == 1:
            temp[h // 2, x] = out[h - 1, x]
    return temp


def haar_inverse(coeffs: np.ndarray):
    h, w = coeffs.shape
    temp = np.zeros_like(coeffs)
    # Inverse column
    for x in range(w):
        for y in range(h // 2):
            a = coeffs[y, x]
            d = coeffs[y + h // 2, x]
            temp[2 * y, x] = a + d
            temp[2 * y + 1, x] = a - d
    out = np.zeros_like(coeffs)
    # Inverse row
    for y in range(h):
        for x in range(w // 2):
            a = temp[y, x]
            d = temp[y, x + w // 2]
            out[y, 2 * x] = a + d
            out[y, 2 * x + 1] = a - d
    return out


def compress(img: np.ndarray, threshold: float = 5.0):
    original_shape = img.shape
    coeffs = haar_forward(img)
    # simple compression: zero small coefficients
    compressed = coeffs.copy()
    compressed[np.abs(compressed) < threshold] = 0
    return compressed, original_shape


def decompress(compressed: np.ndarray, shape):
    recon = haar_inverse(compressed)
    h, w = shape
    # Crop back to original size (in case of padding)
    return np.clip(recon[:h, :w], 0, 255).astype(np.float32)


