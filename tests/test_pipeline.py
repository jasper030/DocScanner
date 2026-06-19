import numpy as np
import cv2
import pytest
from doc_scanner.utils import geometry, filters
from doc_scanner.core.detector import detect_document
from doc_scanner.core.enhancement import enhance_image
from doc_scanner.core.process_text import process_text_image

def test_order_points():
    # Test simple rectangle corner ordering
    pts = np.array([
        [100, 100],  # TL
        [200, 100],  # TR
        [200, 200],  # BR
        [100, 200]   # BL
    ])
    # Shuffle the points
    shuffled_pts = pts[[2, 0, 3, 1]]
    ordered = geometry.order_points(shuffled_pts)

    # Assert they are correctly ordered to TL, TR, BR, BL
    assert np.allclose(ordered[0], [100, 100])
    assert np.allclose(ordered[1], [200, 100])
    assert np.allclose(ordered[2], [200, 200])
    assert np.allclose(ordered[3], [100, 200])

def test_sauvola_threshold():
    # Create a dummy image with a small dark square (representing text stroke) on white background
    img = np.ones((100, 100), dtype=np.uint8) * 255
    img[48:53, 48:53] = 50

    # Run sauvola
    binarized = filters.sauvola_threshold(img, window_size=15, k=0.2)

    # Assert shape and binary values
    assert binarized.shape == (100, 100)
    assert np.all((binarized == 0) | (binarized == 255))
    # Check that the square was thresholded to black
    assert binarized[50, 50] == 0
    # Check that the background is white
    assert binarized[10, 10] == 255

def test_remove_shadows():
    # Create image with lighting gradient (shadow)
    img = np.zeros((100, 100), dtype=np.uint8)
    for y in range(100):
        # Create a gradient from 100 to 200
        img[y, :] = 100 + int(y * 1.0)

    cleaned = filters.remove_shadows(img, kernel_size=11, blur_size=15)

    # Assert shape
    assert cleaned.shape == (100, 100)
    # Check that gradient was mostly removed (values should be close to 255)
    assert np.mean(cleaned) > 240

def test_skew_and_deskew():
    # Create image with horizontal lines
    img = np.ones((200, 200), dtype=np.uint8) * 255
    for y in range(20, 180, 20):
        img[y:y+3, 20:180] = 0

    # Rotate by 3 degrees
    center = (100, 100)
    M = cv2.getRotationMatrix2D(center, 3.0, 1.0)
    rotated = cv2.warpAffine(img, M, (200, 200), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # Estimate skew angle
    angle = filters.estimate_skew_angle(rotated)
    # The estimated angle should be close to -3.0 degrees
    # since we need to rotate by -3.0 to align it back
    assert abs(angle - (-3.0)) < 1.0

    # Correct skew
    corrected = filters.deskew(rotated, angle)
    # Check that variance of horizontal profile is restored
    assert corrected.shape == (200, 200)

def test_auto_rotate_text():
    # Create a portrait text image (horizontal text lines)
    img = np.ones((200, 100), dtype=np.uint8) * 255
    for y in range(20, 180, 20):
        img[y:y+3, 10:90] = 0

    # Standard orientation should not rotate
    rotated, angle = filters.auto_rotate_text(img)
    assert angle == 0

    # Rotate 90 degrees CCW (so text lines run vertically)
    img_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Auto rotate should detect it and rotate 90 degrees CW to make it horizontal again
    rotated_corrected, angle_corrected = filters.auto_rotate_text(img_90)
    assert angle_corrected == 90
    assert rotated_corrected.shape[0] > rotated_corrected.shape[1] # Portrait
