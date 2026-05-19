import cv2

def resize_keep_aspect(image, width=None, height=None):
    """
    Resizes an image while maintaining its aspect ratio.
    """
    h, w = image.shape[:2]
    if width is None and height is None:
        return image, 1.0

    if width is not None:
        ratio = width / float(w)
        new_h = int(h * ratio)
        new_size = (width, new_h)
    else:
        ratio = height / float(h)
        new_w = int(w * ratio)
        new_size = (new_w, height)

    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, ratio
