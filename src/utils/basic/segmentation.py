from scipy.ndimage import binary_fill_holes
from skimage.measure import label


def get_label_image_from_outline(outline_image):
    binary = binary_fill_holes(outline_image)
    binary[outline_image != 0] = 0
    labeled = measure.label(binary)
    return labeled
