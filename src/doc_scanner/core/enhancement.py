import cv2
import numpy as np

from doc_scanner import types
from doc_scanner.utils import filters

def denoise_image(image: types.Image, method: str = "gaussian", kernel_size: int = 5) -> types.Image:
    """
    Apply denoising to the image.
    Methods: 'gaussian', 'median', 'bilateral'
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        return cv2.medianBlur(image, kernel_size)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image

def enhance_image(
    image: types.Image,
    denoise_method: str = "gaussian",
    denoise_kernel: int = 5,
    use_clahe: bool = True,
    use_sharpen: bool = False
) -> types.Image:
    """
    Complete enhancement pipeline.
    """
    processed = image.copy()

    # 1. Denoise
    if denoise_method:
        processed = denoise_image(processed, method=denoise_method, kernel_size=denoise_kernel)

    # 2. Contrast Enhancement
    if use_clahe:
        processed = filters.apply_clahe(processed)

    # 3. Sharpening
    if use_sharpen:
        processed = filters.sharpen(processed)

    return processed
