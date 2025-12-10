import numpy as np


def compress(img: np.ndarray):
    """Simple LZW compression on flattened bytes."""
    data = img.astype(np.uint8).tobytes()
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    w = b""
    codes = []
    for c in data:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = bytes([c])
    if w:
        codes.append(dictionary[w])
    return codes, dictionary


def decompress(codes, shape):
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    result = bytearray()
    w = bytes([codes[0]])
    result += w
    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[:1]
        else:
            raise ValueError("Bad LZW code.")
        result += entry
        dictionary[dict_size] = w + entry[:1]
        dict_size += 1
        w = entry
    arr = np.frombuffer(result, dtype=np.uint8).reshape(shape)
    return arr.astype(np.float32)


