import cv2
from dataclasses import dataclass

from doc_scanner import types
from doc_scanner.utils import filters

@dataclass
class TextProcessingResult:
    otsu: types.Image
    adaptive: types.Image
    cleaned: types.Image
    sauvola: types.Image

def binarize_for_text(image: types.Image, method: str = "sauvola", remove_shadows: bool = True) -> types.Image:
    """
    Binarize an image optimized for text readability and OCR accuracy.
    Methods: 'sauvola', 'adaptive', 'otsu'
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Remove shadows first to normalize background
    if remove_shadows:
        gray = filters.remove_shadows(gray)

    if method == "otsu":
        return filters.otsu_threshold(gray)
    elif method == "adaptive":
        # Calculate dynamic block size based on image dimensions
        h, w = gray.shape[:2]
        block_size = int(max(h, w) / 30)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)
        return filters.adaptive_threshold(gray, block_size=block_size, c=10)
    else:  # Sauvola (default)
        h, w = gray.shape[:2]
        window_size = int(max(h, w) / 45)
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(5, window_size)
        return filters.sauvola_threshold(gray, window_size=window_size, k=0.15)

def clean_text_image(image: types.Image, kernel_size: int = 2) -> types.Image:
    """
    Apply morphological cleaning to a binarized text image to remove fine noise spurs.
    """
    # Small opening to remove tiny isolated salt noise
    cleaned = filters.apply_morphology(image, kernel_size=(kernel_size, kernel_size), op=cv2.MORPH_OPEN)
    return cleaned

def process_text_image(image: types.Image) -> TextProcessingResult:
    """
    Process image to extract text using multiple methods, including advanced Sauvola binarization.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Standard Otsu threshold
    otsu_img = filters.otsu_threshold(gray)

    # 2. Dynamic Adaptive threshold
    h, w = gray.shape[:2]
    block_size = int(max(h, w) / 30)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    adaptive_img = filters.adaptive_threshold(gray, block_size=block_size, c=10)

    # 3. Sauvola threshold (highly optimized for OCR)
    sauvola_img = binarize_for_text(gray, method="sauvola", remove_shadows=True)

    # Cleaned Sauvola version (default cleaned output)
    cleaned = clean_text_image(sauvola_img, kernel_size=2)

    return TextProcessingResult(
        otsu=otsu_img,
        adaptive=adaptive_img,
        cleaned=cleaned,
        sauvola=sauvola_img
    )
