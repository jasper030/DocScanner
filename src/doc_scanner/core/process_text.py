import cv2
import numpy as np

from numpy.typing import NDArray
from doc_scanner.utils import filters

def process_text_image(image: NDArray[np.uint8]):
    otsu_img = filters.otsu_threshold(image)
    adaptive_img = filters.adaptive_threshold(image)

    kernel = np.ones((3,3), np.uint8)
    open_img, close_img = filters.apply_morphology(image, kernel)

    return otsu_img, adaptive_img, open_img, close_img
