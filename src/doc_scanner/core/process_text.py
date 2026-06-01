import cv2
import numpy as np
from dataclasses import dataclass

from doc_scanner import types
from doc_scanner.utils import filters

@dataclass
class TextProcessingResult:
    otsu: types.Image
    adaptive: types.Image
    cleaned: types.Image

def binarize_for_text(image: types.Image, method: str = "adaptive") -> types.Image:
    """
    Binarize an image optimized for text readability.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "otsu":
        return filters.otsu_threshold(image)
    else:
        # Adaptive is usually better for varying lighting
        return filters.adaptive_threshold(image)

def clean_text_image(image: types.Image, kernel_size: int = 2) -> types.Image:
    """
    Apply morphological cleaning to a binarized text image.
    """
    # Small opening to remove noise
    cleaned = filters.apply_morphology(image, kernel_size=(kernel_size, kernel_size), op=cv2.MORPH_OPEN)
    return cleaned

def process_text_image(image: types.Image) -> TextProcessingResult:
    """
    Process image to extract text using multiple methods.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    otsu_img = filters.otsu_threshold(gray)
    adaptive_img = filters.adaptive_threshold(gray)
    
    # Cleaned adaptive version
    cleaned = clean_text_image(adaptive_img)

    return TextProcessingResult(
        otsu=otsu_img,
        adaptive=adaptive_img,
        cleaned=cleaned
    )
