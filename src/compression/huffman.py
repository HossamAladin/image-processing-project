import heapq
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


class _Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_tree(data: np.ndarray) -> _Node:
    freq = defaultdict(int)
    for val in data.flatten():
        freq[int(val)] += 1
    heap = [(_Node(f, s)) for s, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # Edge case: single symbol
        only = heapq.heappop(heap)
        return _Node(only.freq, None, left=only, right=None)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = _Node(a.freq + b.freq, None, a, b)
        heapq.heappush(heap, parent)
    return heap[0]


def _generate_codes(node: _Node, prefix: str, codes: Dict[int, str]) -> None:
    if node.symbol is not None:
        codes[node.symbol] = prefix or "0"
        return
    if node.left:
        _generate_codes(node.left, prefix + "0", codes)
    if node.right:
        _generate_codes(node.right, prefix + "1", codes)


def compress(img: np.ndarray) -> Tuple[str, Dict[int, str]]:
    """Return bitstring and codebook."""
    root = _build_tree(img.astype(np.uint8))
    codes: Dict[int, str] = {}
    _generate_codes(root, "", codes)
    bits = "".join(codes[int(v)] for v in img.flatten())
    return bits, codes


def decompress(bits: str, codes: Dict[int, str], shape) -> np.ndarray:
    """Reconstruct image from bitstring and codebook."""
    inv = {v: k for k, v in codes.items()}
    decoded_vals = []
    buffer = ""
    for b in bits:
        buffer += b
        if buffer in inv:
            decoded_vals.append(inv[buffer])
            buffer = ""
    arr = np.array(decoded_vals, dtype=np.uint8).reshape(shape)
    return arr.astype(np.float32)


