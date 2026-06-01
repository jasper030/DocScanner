import cv2
import numpy as np

from doc_scanner import types

def otsu_threshold(image: types.Image) -> types.Image:
    """
    Apply Otsu threshold and return thresholded image.
    If image is BGR, it will be converted to grayscale first.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def adaptive_threshold(image: types.Image, block_size: int = 11, c: int = 2) -> types.Image:
    """
    Apply adaptive threshold and return thresholded image.
    If image is BGR, it will be converted to grayscale first.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )

def apply_morphology(
    image: types.Image,
    kernel_size: tuple[int, int] = (3, 3),
    op: int = cv2.MORPH_OPEN
) -> types.Image:
    """
    Apply morphological operation to the image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, op, kernel)

def apply_clahe(
    image: types.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8)
) -> types.Image:
    """
    Apply CLAHE for local contrast enhancement.
    Supports both grayscale and BGR (enhances V channel in HSV).
    """
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        v = clahe.apply(v)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def sharpen(image: types.Image) -> types.Image:
    """
    Sharpen image using a kernel filter.
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, kernel)
