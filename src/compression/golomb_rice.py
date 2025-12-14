import numpy as np


def _encode_value(val: int, k: int) -> str:
    m = 1 << k
    q = val // m
    r = val % m
    return "1" * q + "0" + format(r, f"0{k}b")


def _decode_stream(bits: str, k: int):
    m = 1 << k
    i = 0
    values = []
    while i < len(bits):
        q = 0
        while i < len(bits) and bits[i] == "1":
            q += 1
            i += 1
        if i >= len(bits):
            break
        i += 1  # skip zero separator
        if i + k > len(bits):
            break
        r = int(bits[i : i + k], 2)
        i += k
        values.append(q * m + r)
    return values


def compress(img: np.ndarray, k: int = 2):
    data = img.astype(np.uint8).flatten()
    bits = "".join(_encode_value(int(v), k) for v in data)
    return bits, k, img.shape


def decompress(bits: str, k: int, shape):
    values = _decode_stream(bits, k)
    arr = np.array(values, dtype=np.uint8).reshape(shape)
    return arr.astype(np.float32)


