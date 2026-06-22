"""
Black Tape Line Detection & Following — ESP32 Camera Stream
Detects a black tape line on the floor and provides steering guidance.
Publishes the annotated stream on http://localhost:5002/stream
"""

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

# ── Configuration ────────────────────────────────────────────────────
STREAM_URL = "http://10.200.7.27/stream"
PORT = 5002

# CLAHE (brightness enhancement)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)

# Blur
MEDIAN_BLUR_K = 7
GAUSSIAN_BLUR_K = (7, 7)

# Morphology kernel sizes
ERODE_K = 5
ERODE_ITER = 2
DILATE_K = 7
DILATE_ITER = 2
CLOSE_K = 15
OPEN_K = 11

# Detection thresholds
MIN_CONTOUR_AREA = 2000
MIN_STRIP_PIXELS = 10
NUM_SLICES = 15
CENTER_DEADZONE = 25  # pixels from center considered "straight"
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Black Tape Line Detector")


def detect_tape(frame: np.ndarray) -> np.ndarray:
    """Process a single frame: detect the black tape and annotate the image."""
    h, w = frame.shape[:2]

    # --- Pre-processing ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    enhanced = clahe.apply(gray)

    blurred = cv2.medianBlur(enhanced, MEDIAN_BLUR_K)
    blurred = cv2.GaussianBlur(blurred, GAUSSIAN_BLUR_K, 0)

    # --- Thresholding (Otsu auto-selects the optimal value) ---
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Morphological noise removal ---
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_K, ERODE_K))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_K, CLOSE_K))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_K, OPEN_K))

    mask = cv2.erode(mask, k_erode, iterations=ERODE_ITER)
    mask = cv2.dilate(mask, k_dilate, iterations=DILATE_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # --- Contour detection ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    # --- Build the display frame (enhanced so it's visible) ---
    display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.line(display, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)  # center ref

    if not large_contours:
        _draw_status(display, "NO TAPE FOUND", (0, 0, 255))
    else:
        tape = max(large_contours, key=cv2.contourArea)
        cv2.drawContours(display, [tape], -1, (0, 255, 0), 3)

        # Row-by-row centroid tracking
        path = _track_path(mask, tape, h, w)

        for pt in path:
            cv2.circle(display, pt, 5, (0, 255, 255), -1)

        if len(path) >= 2:
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], False, (255, 0, 255), 3)
            _draw_steering(display, path[-1][0], w, h)
        else:
            _draw_status(display, "TAPE PARTIALLY VISIBLE", (0, 165, 255))

    # --- Side-by-side: annotated frame | binary mask ---
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack((display, mask_bgr))


def _track_path(mask: np.ndarray, contour, h: int, w: int) -> list:
    """Scan horizontal slices and return the tape center at each row."""
    tape_only = np.zeros_like(mask)
    cv2.drawContours(tape_only, [contour], -1, 255, -1)

    points = []
    slice_h = h // NUM_SLICES
    for i in range(NUM_SLICES):
        y0 = i * slice_h
        y1 = y0 + slice_h
        strip = tape_only[y0:y1, :]
        cols = np.where(strip > 0)[1]
        if len(cols) > MIN_STRIP_PIXELS:
            points.append((int(np.mean(cols)), (y0 + y1) // 2))
    return points


def _draw_steering(frame: np.ndarray, tape_x: int, w: int, h: int):
    """Draw a steering arrow and direction label at the bottom of the frame."""
    center_x = w // 2
    deviation = tape_x - center_x

    cv2.arrowedLine(frame, (center_x, h - 30), (tape_x, h - 30),
                    (0, 0, 255), 3, tipLength=0.3)

    if abs(deviation) < CENTER_DEADZONE:
        direction, color = "CENTER", (0, 255, 0)
    elif deviation > 0:
        direction, color = "RIGHT", (0, 0, 255)
    else:
        direction, color = "LEFT", (0, 0, 255)

    cv2.putText(frame, f"Steer: {direction} ({deviation:+d}px)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_status(frame: np.ndarray, text: str, color: tuple):
    """Draw a status message on the frame."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ── Streaming ────────────────────────────────────────────────────────

def generate_frames():
    """Generator that yields MJPEG frames from the processed stream."""
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"Error: Could not open stream at {STREAM_URL}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            result = detect_tape(frame)
            ok, buffer = cv2.imencode('.jpg', result)
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.get('/stream')
def video_feed():
    """MJPEG video stream endpoint."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == '__main__':
    print("Starting Black Tape Line Detector...")
    print(f"Stream available at: http://localhost:{PORT}/stream")
    uvicorn.run(app, host='0.0.0.0', port=PORT)
