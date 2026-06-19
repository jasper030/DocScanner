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
    Highly robust document detection using multiple preprocessing passes,
    sub-pixel corner refinement, and fallback corner estimation.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image not found at: {image_path}")

    original = image.copy()
    resized, ratio = resize_keep_aspect(image, width=config.RESIZE_WIDTH)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    image_area = resized.shape[0] * resized.shape[1]

    # Preprocessing strategies
    # 1. CLAHE + Bilateral + Canny (Best for faint boundaries on light backgrounds)
    # 2. Bilateral + Canny (Good for edges, bad for noise)
    # 3. Gaussian + Canny (Standard)
    # 4. Otsu Thresholding (Good for high contrast documents)
    # 5. Adaptive Thresholding (Good for uneven lighting)

    def clahe_canny(g):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(g)
        return cv2.Canny(cv2.bilateralFilter(clahe_img, 9, 75, 75), 30, 150)

    passes = [
        ("CLAHE+Bilateral+Canny", clahe_canny),
        ("Bilateral+Canny", lambda g: cv2.Canny(cv2.bilateralFilter(g, 9, 75, 75), 50, 200)),
        ("Gaussian+Canny", lambda g: cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 150)),
        ("Otsu", lambda g: cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("Adaptive", lambda g: cv2.adaptiveThreshold(cv2.medianBlur(g, 5), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2))
    ]

    document_contour = None
    last_processed = None

    for _, preprocess_fn in passes:
        edged = preprocess_fn(gray)

        # Connect fragmented lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
        last_processed = processed

        # Try finding external contours first (cleaner) then list (more thorough)
        for mode in [cv2.RETR_EXTERNAL, cv2.RETR_LIST]:
            contours, _ = cv2.findContours(processed.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image_area * 0.05 or area > image_area * 0.99:
                    continue

                peri = cv2.arcLength(contour, True)

                # Strategy A: Strict Polygon Approximation
                for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        document_contour = approx
                        break
                if document_contour is not None:
                    break

                # Strategy B: Convex Hull Approximation (if the document has ragged edges)
                hull = cv2.convexHull(contour)
                hull_peri = cv2.arcLength(hull, True)
                for eps in [0.01, 0.02, 0.03, 0.05]:
                    approx = cv2.approxPolyDP(hull, eps * hull_peri, True)
                    if len(approx) == 4:
                        document_contour = approx
                        break
                if document_contour is not None:
                    break

            if document_contour is not None:
                break
        if document_contour is not None:
            break

    if document_contour is None:
        # Final Desperate Fallback: Take the largest candidate and FORCE it to 4 points
        # based on extreme coordinates (bounding box-ish logic)
        contours, _ = cv2.findContours(last_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > image_area * 0.1:
                # Get the 4 extreme points
                pts = largest.reshape(-1, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)] # TL
                rect[2] = pts[np.argmax(s)] # BR
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)] # TR
                rect[3] = pts[np.argmax(diff)] # BL
                document_contour = rect.reshape(4, 1, 2).astype(np.int32)

    if document_contour is None:
        raise RuntimeError("Detection failed. Please ensure the document is clearly visible against the background.")

    # Scale back to original
    document_contour_original = document_contour.reshape(4, 2) / ratio

    # Corner Refinement (Sub-pixel accuracy)
    try:
        corners = document_contour_original.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        # Refine corners in the original high-resolution image
        refined_corners = cv2.cornerSubPix(gray_orig, corners, (11, 11), (-1, -1), criteria)
        # Ensure points remain valid
        if refined_corners is not None and len(refined_corners) == 4:
            document_contour_original = refined_corners
    except Exception:
        # Fall back to unrefined corners if sub-pixel refinement fails
        pass

    warped = four_point_transform(original, document_contour_original)

    return DetectionResult(
        original=original,
        resized=resized,
        edged=last_processed,
        contour=document_contour,
        warped=warped
    )
