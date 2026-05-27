import cv2
import numpy as np


def gaussian_denoise(img, kernel_size=(5, 5)):
    """
    Gaussian Blur for noise reduction
    """
    return cv2.GaussianBlur(img, kernel_size, 0)


def median_denoise(img, kernel_size=5):
    """
    Median Blur for salt-and-pepper noise reduction
    """
    return cv2.medianBlur(img, kernel_size)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE for local contrast enhancement
    """

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(gray)


def sharpen(img):
    """
    Sharpen image using kernel filter
    """

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(img, -1, kernel)


def enhance_image(img,
                  denoise_method="gaussian",
                  use_sharpen=False):
    """
    Complete enhancement pipeline
    """

    # Denoising
    if denoise_method == "gaussian":
        img = gaussian_denoise(img)

    elif denoise_method == "median":
        img = median_denoise(img)

    # CLAHE enhancement
    img = apply_clahe(img)

    # Optional sharpening
    if use_sharpen:
        img = sharpen(img)

    return img
