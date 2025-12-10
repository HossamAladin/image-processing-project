import numpy as np


def _build_probs(data: np.ndarray):
    counts = np.bincount(data.flatten(), minlength=256).astype(np.float64)
    total = counts.sum()
    probs = counts / (total + 1e-8)
    cdf = np.cumsum(probs)
    cdf = np.insert(cdf, 0, 0.0)  # prepend 0
    return probs, cdf


def compress(img: np.ndarray):
    data = img.astype(np.uint8).flatten()
    probs, cdf = _build_probs(img.astype(np.uint8))
    low, high = 0.0, 1.0
    for symbol in data:
        range_ = high - low
        high = low + range_ * cdf[symbol + 1]
        low = low + range_ * cdf[symbol]
    code = (low + high) / 2.0
    return code, probs, img.shape


def decompress(code: float, probs, shape):
    cdf = np.cumsum(probs)
    cdf = np.insert(cdf, 0, 0.0)
    data = []
    for _ in range(shape[0] * shape[1]):
        for symbol in range(256):
            if cdf[symbol] <= code < cdf[symbol + 1]:
                data.append(symbol)
                low, high = cdf[symbol], cdf[symbol + 1]
                code = (code - low) / (high - low + 1e-12)
                break
    arr = np.array(data, dtype=np.uint8).reshape(shape)
    return arr.astype(np.float32)


