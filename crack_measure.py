import cv2
import numpy as np


def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Morphological thinning to reduce a crack mask to a 1-pixel-wide centerline."""
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


def calculate_crack_length(opencv_img: np.ndarray):
    """
    Takes a BGR OpenCV image, segments dark microcracks under soft focus or uneven
    lighting, extracts the 1-pixel centerline skeleton, and returns:
        (length_in_pixels: float, annotated_bgr_image: np.ndarray)
    """
    overlay = opencv_img.copy()
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

    # 1. Local contrast enhancement + blur to handle soft ESP32-CAM focus
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 2. Adaptive inverse threshold to isolate dark fissures on lighter surfaces
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=12,
    )

    # 3. Clean small speckles and bridge tiny gaps along the fissure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Keep elongated crack contours (reject small dots and large shadow blocks)
    h, w = gray.shape
    img_area = h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 45 or area > img_area * 0.25:
            continue

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        aspect_ratio = max(rw, rh) / min(rw, rh)

        # Cracks are elongated structures (aspect ratio >= 2.2)
        if aspect_ratio >= 2.2:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            valid_contours.append(c)

    # 5. Measure true pixel length along the 1-pixel skeleton centerline
    skeleton = _skeletonize(clean_mask)
    length_px = float(cv2.countNonZero(skeleton))

    # Fallback to half-perimeter if skeletonization leaves 0 pixels on very thin segments
    if length_px == 0.0 and valid_contours:
        length_px = float(sum(cv2.arcLength(c, closed=True) * 0.5 for c in valid_contours))

    # 6. Draw contour boundary + bright green centerline on output preview
    cv2.drawContours(overlay, valid_contours, -1, (0, 200, 255), 1)
    overlay[skeleton > 0] = (0, 255, 80)

    return round(length_px, 2), overlay