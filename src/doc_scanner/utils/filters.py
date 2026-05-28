import cv2
import numpy as np

def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu threshold and return thresholded image
    """
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive threshold and return thresholded image
    """
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

def apply_morphology(
        image: np.ndarray,
        kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply morphology and return opened and closed image
    """
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return opening, closing
