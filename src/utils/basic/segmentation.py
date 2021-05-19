from typing import Tuple

from scipy.ndimage import binary_fill_holes
from skimage.measure import label

import numpy as np

from numpy import ndarray


def get_label_image_from_outline(outline_image: ndarray) -> ndarray:
    binary = binary_fill_holes(outline_image)
    binary[outline_image != 0] = 0
    labeled = label(binary)
    return labeled


def pad_image(image: ndarray, size: Tuple[int]) -> ndarray:
    padded_img = np.zeros(size)
    img_x, img_y = image.shape
    pimg_x, pimg_y = padded_img.shape

    pimg_xmid = pimg_x // 2 + pimg_x % 2
    pimg_ymid = pimg_y // 2 + pimg_y % 2

    img_xhalf = img_x // 2
    img_yhalf = img_y // 2

    padded_img[
        pimg_xmid - img_xhalf : pimg_xmid + (img_x - img_xhalf),
        pimg_ymid - img_yhalf : pimg_ymid + (img_y - img_yhalf),
    ] = image

    return padded_img
