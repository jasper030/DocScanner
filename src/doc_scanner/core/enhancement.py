import cv2

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
    use_sharpen: bool = False,
    use_unsharp: bool = True,
    remove_shadows: bool = True,
    auto_orient: bool = True,
    deskew: bool = True
) -> types.Image:
    """
    Complete enhancement pipeline for high-quality scanned document output.
    """
    processed = image.copy()

    # 1. Auto Orientation & Deskewing
    if auto_orient:
        processed, _ = filters.auto_rotate_text(processed)

    if deskew:
        skew_angle = filters.estimate_skew_angle(processed)
        # Only deskew if skew is significant (e.g. > 0.1 degrees)
        if abs(skew_angle) > 0.1:
            processed = filters.deskew(processed, skew_angle)

    # 2. Shadow Removal / Background Normalization
    if remove_shadows:
        processed = filters.remove_shadows(processed)

    # 3. Denoise
    if denoise_method:
        processed = denoise_image(processed, method=denoise_method, kernel_size=denoise_kernel)

    # 4. Contrast Enhancement
    if use_clahe:
        # Applying CLAHE after shadow removal boosts local details
        processed = filters.apply_clahe(processed)

    # 5. Sharpening
    if use_sharpen:
        if use_unsharp:
            processed = filters.unsharp_mask(processed, sigma=1.0, strength=1.5)
        else:
            processed = filters.sharpen(processed)

    return processed
