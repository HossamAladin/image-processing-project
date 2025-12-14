import math
import numpy as np

_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


def _alpha(u: int) -> float:
    return 1 / math.sqrt(2) if u == 0 else 1.0


def dct2(block: np.ndarray) -> np.ndarray:
    block = block - 128.0
    out = np.zeros((8, 8), dtype=np.float32)
    for u in range(8):
        for v in range(8):
            s = 0.0
            for x in range(8):
                for y in range(8):
                    s += block[x, y] * math.cos(((2 * x + 1) * u * math.pi) / 16) * math.cos(
                        ((2 * y + 1) * v * math.pi) / 16
                    )
            out[u, v] = 0.25 * _alpha(u) * _alpha(v) * s
    return out


def idct2(block: np.ndarray) -> np.ndarray:
    out = np.zeros((8, 8), dtype=np.float32)
    for x in range(8):
        for y in range(8):
            s = 0.0
            for u in range(8):
                for v in range(8):
                    s += _alpha(u) * _alpha(v) * block[u, v] * math.cos(
                        ((2 * x + 1) * u * math.pi) / 16
                    ) * math.cos(((2 * y + 1) * v * math.pi) / 16)
            out[x, y] = 0.25 * s
    return out + 128.0


def _process_blocks(gray: np.ndarray, fn):
    h, w = gray.shape
    padded_h = (h + 7) // 8 * 8
    padded_w = (w + 7) // 8 * 8
    padded = np.zeros((padded_h, padded_w), dtype=np.float32)
    padded[:h, :w] = gray
    blocks = []
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            block = padded[i : i + 8, j : j + 8]
            blocks.append(fn(block))
    return blocks, (h, w), (padded_h, padded_w)


def compress(gray: np.ndarray):
    blocks, orig_shape, padded_shape = _process_blocks(gray, dct2)
    quantized = [np.round(b / _Q) for b in blocks]
    return quantized, orig_shape, padded_shape


def decompress(quantized_blocks, orig_shape, padded_shape):
    padded = np.zeros(padded_shape, dtype=np.float32)
    idx = 0
    for i in range(0, padded_shape[0], 8):
        for j in range(0, padded_shape[1], 8):
            deq = quantized_blocks[idx] * _Q
            block = idct2(deq)
            padded[i : i + 8, j : j + 8] = block
            idx += 1
    h, w = orig_shape
    return np.clip(padded[:h, :w], 0, 255)


