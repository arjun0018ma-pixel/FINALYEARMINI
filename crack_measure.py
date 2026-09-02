import cv2
import numpy as np
from skimage.morphology import skeletonize

def calculate_crack_length(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > w * 2 and cv2.contourArea(cnt) > 10:
            filtered.append(cnt)

    if not filtered:
        return 0.0, image.copy()

    total_length = sum(cv2.arcLength(cnt, False) for cnt in filtered)
    result = image.copy()
    cv2.drawContours(result, filtered, -1, (0, 255, 0), 2)

    return total_length, result

def calculate_crack_length_v3(image, roi=None):
    """
    roi: optional (y1, y2, x1, x2). Leave as None for general use —
    only set it if you're manually cropping out a specific known
    obstruction in one particular photo. Never hardcode a fixed ROI
    for all images; crack location varies per photo.
    """
    work = image.copy()
    # Diagnostic: log input shape and dtype to help debug runtime issues
    try:
        print(f"[crack-length] input shape: {work.shape}, dtype: {work.dtype}")
    except Exception:
        print("[crack-length] input info unavailable")
    if roi is not None:
        y1, y2, x1, x2 = roi
        work = work[y1:y2, x1:x2]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    med = np.median(blur)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(blur, lower, upper)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print(f"[crack-length] contours found: {len(contours)}")

    # FIX: rotated bounding box works at any angle, not just vertical
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
        if rw == 0 or rh == 0:
            continue
        long_side, short_side = max(rw, rh), min(rw, rh)
        if long_side < 15:
            continue
        if long_side / short_side > 3:
            filtered.append(cnt)

    print(f"[crack-length] filtered (passed shape test): {len(filtered)}")
    if not filtered:
        print("[crack-length] no filtered contours — returning zero length")
        return 0.0, image.copy(), None

    mask = np.zeros(closed.shape, dtype=np.uint8)
    cv2.drawContours(mask, filtered, -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask, filtered, -1, 255, thickness=2)

    skeleton = skeletonize(mask > 0)
    ys, xs = np.where(skeleton)
    coords = set(zip(ys.tolist(), xs.tolist()))
    total_length = 0.0
    for (y, x) in coords:
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            if (y+dy, x+dx) in coords:
                total_length += (dy**2 + dx**2) ** 0.5
    total_length /= 2.0

    result = image.copy()
    offset_y, offset_x = (roi[0], roi[2]) if roi is not None else (0, 0)
    sk_ys, sk_xs = np.where(skeleton)
    for y, x in zip(sk_ys, sk_xs):
        cv2.circle(result, (x + offset_x, y + offset_y), 2, (0, 0, 255), -1)

    print(f"[crack-length] final length: {total_length}")
    return total_length, result, skeleton


if __name__ == "__main__":
    img = cv2.imread("your_image.png")

    # Restrict to the region containing just the crack — adjust these
    # to your photo (excludes roof texture, pipes, switch boxes, etc.)
    roi = None

    length, result, skeleton = calculate_crack_length_v3(img, roi=roi)
    print("Total crack length (skeleton, px):", round(length, 2))
    cv2.imwrite("result.png", result)