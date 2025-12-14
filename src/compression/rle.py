import numpy as np


def compress(img: np.ndarray):
    data = img.astype(np.uint8).flatten()
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append((int(data[i - 1]), count))
            count = 1
    encoded.append((int(data[-1]), count))
    return encoded, img.shape


def decompress(encoded, shape):
    values = []
    for val, cnt in encoded:
        values.extend([val] * cnt)
    arr = np.array(values, dtype=np.uint8).reshape(shape)
    return arr.astype(np.float32)


