import os
from typing import Tuple

import numpy as np
from PIL import Image

from .utils import array_to_pil, image_info, load_image_to_array


def load_image(path: str) -> Tuple[np.ndarray, dict]:
    """Load an image and return array + info."""
    arr = load_image_to_array(path)
    info = image_info(arr, file_size=os.path.getsize(path), mode_hint="RGB")
    return arr, info


def save_image(arr: np.ndarray, path: str) -> None:
    """Save numpy image to disk."""
    img = array_to_pil(arr)
    img.save(path)


