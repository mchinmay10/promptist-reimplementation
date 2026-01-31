from PIL import Image
import numpy as np


def make_dummy_image(size=224):
    """
    Create a dummy RGB image for reward testing
    """
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)
