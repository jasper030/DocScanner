import cv2
import numpy as np
from dataclasses import dataclass

from doc_scanner import config
from doc_scanner import types
from doc_scanner.utils.image import resize_keep_aspect
from doc_scanner.utils.geometry import four_point_transform

@dataclass
class DetectionResult:
    original: types.Image
    resized: types.Image
    edged: types.Image
    contour: types.Contour
    warped: types.Image

def detect_document(image_path: str) -> DetectionResult:
    """
    Detects a document in an image and returns the warped result and intermediate steps.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image not found at: {image_path}")

    original = image.copy()

    resized, ratio = resize_keep_aspect(image, width=config.RESIZE_WIDTH)

    # Image Processing Pipeline for Detection
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
    edged = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)

    # Morphology to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    # Contour Detection
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None
    image_area = resized.shape[0] * resized.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.1: # Minimum area threshold
            continue

        peri = cv2.arcLength(contour, True)
        # Try multiple epsilon values for approximation
        for epsilon_ratio in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
            if len(approx) == 4:
                document_contour = approx
                break
        
        if document_contour is not None:
            break

    if document_contour is None:
        raise RuntimeError("Could not find document contour. Try a clearer image or different lighting.")

    # Scale contour back to original image size
    document_contour_original = document_contour.reshape(4, 2) / ratio
    warped = four_point_transform(original, document_contour_original)

    return DetectionResult(
        original=original,
        resized=resized,
        edged=edged,
        contour=document_contour,
        warped=warped
    )
