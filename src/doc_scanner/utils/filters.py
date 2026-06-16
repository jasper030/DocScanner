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

def unsharp_mask(image: types.Image, sigma: float = 1.0, strength: float = 1.5) -> types.Image:
    """
    Sharpen image using unsharp masking (Gaussian blur subtraction).
    Less sensitive to high-frequency noise than a standard 2D kernel.
    """
    if len(image.shape) == 3:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    else:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def remove_shadows(image: types.Image, kernel_size: int = 21, blur_size: int = 25) -> types.Image:
    """
    Remove shadows and uneven illumination using background division.
    Works on both grayscale and color BGR images.
    """
    if len(image.shape) == 3:
        channels = cv2.split(image)
        processed_channels = []
        for ch in channels:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            dilated = cv2.dilate(ch, kernel)
            bg = cv2.medianBlur(dilated, blur_size)
            div = cv2.divide(ch, bg, scale=255)
            processed_channels.append(div)
        return cv2.merge(processed_channels)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(image, kernel)
        bg = cv2.medianBlur(dilated, blur_size)
        return cv2.divide(image, bg, scale=255)

def sauvola_threshold(
    image: types.Image,
    window_size: int = 25,
    k: float = 0.15,
    r: float = 128.0
) -> types.Image:
    """
    Apply Sauvola local adaptive thresholding.
    Optimized implementation using cv2.boxFilter.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_f = image.astype(np.float32)

    # Compute local mean
    mean = cv2.boxFilter(gray_f, -1, (window_size, window_size))

    # Compute local mean of square
    sq_mean = cv2.boxFilter(gray_f * gray_f, -1, (window_size, window_size))

    # Local variance and standard deviation
    var = sq_mean - mean * mean
    var = np.clip(var, 0, None)
    std = np.sqrt(var)

    # Sauvola threshold formula
    threshold = mean * (1.0 + k * (std / r - 1.0))

    # Binarize
    binarized = np.zeros_like(image)
    binarized[gray_f > threshold] = 255
    return binarized

def estimate_skew_angle(image: types.Image) -> float:
    """
    Estimate the skew angle of the document text using projection profile method.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize to speed up calculation
    h_orig, w_orig = gray.shape
    scale = 400.0 / max(h_orig, w_orig)
    resized = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

    # Binarize (white text on black background)
    thresh = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    # Scan angles from -10.0 to 10.0 degrees in steps of 0.25
    angles = np.arange(-10.0, 10.0, 0.25)
    best_angle = 0.0
    max_var = -1.0

    h_res, w_res = thresh.shape
    center = (w_res // 2, h_res // 2)

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (w_res, h_res), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        hp = np.sum(rotated, axis=1) / w_res
        var = np.var(hp)

        if var > max_var:
            max_var = var
            best_angle = angle

    return best_angle

def deskew(image: types.Image, angle: float) -> types.Image:
    """
    Rotate image by a given angle (in degrees) with white background border.
    """
    if abs(angle) < 0.01:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    border_val = (255, 255, 255) if len(image.shape) == 3 else 255
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return corrected

def auto_rotate_text(image: types.Image) -> tuple[types.Image, int]:
    """
    Detect text orientation and rotate the image so text is horizontal (portrait).
    Returns the rotated image and the rotation angle applied (0, 90, 180, 270).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Get a binary image (white text on black background)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    # Calculate horizontal and vertical projection profile variances
    hp = np.sum(thresh, axis=1) / thresh.shape[1]
    vp = np.sum(thresh, axis=0) / thresh.shape[0]

    var_h = np.var(hp)
    var_v = np.var(vp)

    if var_v > var_h:
        # Text lines run vertically. Let's rotate 90 degrees clockwise.
        rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_cw, 90

    return image, 0
