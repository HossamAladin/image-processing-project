import numpy as np


def compress(img: np.ndarray):
    """Bit-plane coding: split into 8 bitplanes."""
    data = img.astype(np.uint8)
    planes = []
    for bit in range(8):
        mask = 1 << bit
        plane = ((data & mask) >> bit).astype(np.uint8)
        planes.append("".join(str(v) for v in plane.flatten()))
    return planes, img.shape


def decompress(planes, shape):
    h, w = shape
    out = np.zeros(h * w, dtype=np.uint8)
    for bit, plane_bits in enumerate(planes):
        plane_array = np.array(list(map(int, plane_bits)), dtype=np.uint8)
        out += (plane_array << bit)
    return out.reshape(shape).astype(np.float32)


