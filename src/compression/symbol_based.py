import numpy as np


def compress(img: np.ndarray):
    """Simple symbol-based coding: top-16 frequent symbols get short codes."""
    data = img.astype(np.uint8).flatten()
    counts = np.bincount(data, minlength=256)
    top16 = np.argsort(counts)[::-1][:16]
    top_set = set(top16.tolist())
    code_map = {}
    for i, sym in enumerate(top16):
        code_map[int(sym)] = "0" + format(i, "04b")  # 5 bits
    bits = []
    for v in data:
        iv = int(v)
        if iv in top_set:
            bits.append(code_map[iv])
        else:
            bits.append("1" + format(iv, "08b"))  # 9 bits fallback
    return "".join(bits), top16, img.shape


def decompress(bits: str, top16, shape):
    top_list = list(top16)
    values = []
    i = 0
    while i < len(bits):
        if bits[i] == "0":
            code = bits[i + 1 : i + 5]
            idx = int(code, 2)
            values.append(top_list[idx])
            i += 5
        else:
            code = bits[i + 1 : i + 9]
            values.append(int(code, 2))
            i += 9
    arr = np.array(values, dtype=np.uint8).reshape(shape)
    return arr.astype(np.float32)


