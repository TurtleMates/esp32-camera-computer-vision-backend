"""
ESP32 Camera — Unified Vision API
Serves 4 video stream endpoints from a single FastAPI server:
  /stream/raw      → Raw camera feed (no processing)
  /stream/human    → YOLO human detection & segmentation
  /stream/signs    → YOLO road sign detection (stop signs & traffic lights)
  /stream/line     → Black tape line follower (OpenCV)
"""

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

# ── Configuration ────────────────────────────────────────────────────
STREAM_URL = "http://192.168.137.75/stream"
PORT = 5000

# YOLO
YOLO_MODEL = "yolov8n-seg.pt"
HUMAN_CONF = 0.5
SIGNS_CONF = 0.4

# Line follower — CLAHE
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)

# Line follower — Blur
MEDIAN_K = 7
GAUSS_K = (7, 7)

# Line follower — Morphology
ERODE_K = 5
ERODE_ITER = 2
DILATE_K = 7
DILATE_ITER = 2
CLOSE_K = 15
OPEN_K = 11

# Line follower — Detection
MIN_CONTOUR_AREA = 2000
MIN_STRIP_PX = 10
NUM_SLICES = 15
CENTER_DEADZONE = 25
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="ESP32 Unified Vision API")

# Allow any web app (on any port/domain) to call these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model once (shared between human & sign detection)
model = YOLO(YOLO_MODEL)

# Warmup: run a dummy inference to pre-compile CUDA kernels.
# Without this, the first real request has a ~2-5s delay.
print("Warming up YOLO model...")
_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model(_dummy, verbose=False)
print("Model ready!")


# =====================================================================
#  Processing Functions
# =====================================================================

def process_raw(frame: np.ndarray) -> np.ndarray:
    """No processing — return the frame as-is."""
    return frame


def process_human(frame: np.ndarray) -> np.ndarray:
    """Detect and segment humans using YOLOv8 (COCO class 0 = person)."""
    # Use model.predict to ensure the classes filter is strictly applied
    results = model.predict(source=frame, classes=[0], conf=HUMAN_CONF, verbose=False)
    return results[0].plot()


def process_signs(frame: np.ndarray) -> np.ndarray:
    """Detect road signs using YOLOv8 (class 9 = traffic light, 11 = stop sign)."""
    # Use model.predict to ensure the classes filter is strictly applied
    results = model.predict(source=frame, classes=[9, 11], conf=SIGNS_CONF, verbose=False)
    return results[0].plot()


def process_line(frame: np.ndarray) -> np.ndarray:
    """Detect a black tape line on the floor and annotate steering guidance."""
    h, w = frame.shape[:2]

    # Pre-processing: enhance dark ESP32 image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enhanced = clahe.apply(gray)

    blurred = cv2.medianBlur(enhanced, MEDIAN_K)
    blurred = cv2.GaussianBlur(blurred, GAUSS_K, 0)

    # Otsu auto-threshold
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological noise removal
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_K, ERODE_K))
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_K, CLOSE_K))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_K, OPEN_K))

    mask = cv2.erode(mask, ke, iterations=ERODE_ITER)
    mask = cv2.dilate(mask, kd, iterations=DILATE_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)

    # Find the tape contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.line(display, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)

    if not large:
        _put_text(display, "NO TAPE FOUND", (0, 0, 255))
    else:
        tape = max(large, key=cv2.contourArea)
        cv2.drawContours(display, [tape], -1, (0, 255, 0), 3)

        path = _track_path(mask, tape, h, w)
        for pt in path:
            cv2.circle(display, pt, 5, (0, 255, 255), -1)

        if len(path) >= 2:
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], False, (255, 0, 255), 3)
            _draw_steering(display, path[-1][0], w, h)
        else:
            _put_text(display, "TAPE PARTIALLY VISIBLE", (0, 165, 255))

    # Side-by-side: annotated | binary mask
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack((display, mask_bgr))


# ── Line-follower helpers ────────────────────────────────────────────

def _track_path(mask, contour, h, w):
    tape_only = np.zeros_like(mask)
    cv2.drawContours(tape_only, [contour], -1, 255, -1)
    points = []
    sh = h // NUM_SLICES
    for i in range(NUM_SLICES):
        y0, y1 = i * sh, (i + 1) * sh
        cols = np.where(tape_only[y0:y1, :] > 0)[1]
        if len(cols) > MIN_STRIP_PX:
            points.append((int(np.mean(cols)), (y0 + y1) // 2))
    return points


def _draw_steering(frame, tape_x, w, h):
    cx = w // 2
    dev = tape_x - cx
    cv2.arrowedLine(frame, (cx, h - 30), (tape_x, h - 30), (0, 0, 255), 3, tipLength=0.3)
    if abs(dev) < CENTER_DEADZONE:
        label, color = "CENTER", (0, 255, 0)
    else:
        label, color = ("RIGHT" if dev > 0 else "LEFT"), (0, 0, 255)
    cv2.putText(frame, f"Steer: {label} ({dev:+d}px)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _put_text(frame, text, color):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# =====================================================================
#  Shared Camera Reader (single connection to ESP32)
# =====================================================================

class CameraReader:
    """Reads frames from the ESP32 in a background thread.
    Only one connection to the camera is ever open, so switching
    endpoints is instant."""

    def __init__(self, url: str):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            print(f"Error: Could not open stream at {self.url}")
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.url)
                continue
            with self.lock:
                self.frame = frame
        cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)


# Start the shared camera reader at startup
camera = CameraReader(STREAM_URL)


def _stream(processor):
    """Generic MJPEG stream generator. Grabs the latest frame from
    the shared camera reader and applies the given processor."""
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            
            result = processor(frame)
            ok, buf = cv2.imencode('.jpg', result)
            if ok:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            
            # Throttle the output stream to ~10 FPS to reduce lag
            time.sleep(0.1)
    except GeneratorExit:
        pass


# =====================================================================
#  API Endpoints
# =====================================================================

@app.get('/', response_class=HTMLResponse)
def index():
    """Landing page listing all available streams."""
    return """
    <html>
    <head><title>ESP32 Vision API</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
        <h1>ESP32 Unified Vision API</h1>
        <ul style="line-height: 2.2;">
            <li><a href="/stream/raw">/stream/raw</a> — Raw camera feed</li>
            <li><a href="/stream/human">/stream/human</a> — Human detection &amp; segmentation</li>
            <li><a href="/stream/signs">/stream/signs</a> — Road sign detection</li>
            <li><a href="/stream/line">/stream/line</a> — Black tape line follower</li>
        </ul>
    </body>
    </html>
    """


@app.get('/stream/raw')
def stream_raw():
    """Raw camera feed — no processing."""
    return StreamingResponse(_stream(process_raw),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get('/stream/human')
def stream_human():
    """Human detection & segmentation."""
    return StreamingResponse(_stream(process_human),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get('/stream/signs')
def stream_signs():
    """Road sign detection (stop signs & traffic lights)."""
    return StreamingResponse(_stream(process_signs),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get('/stream/line')
def stream_line():
    """Black tape line follower."""
    return StreamingResponse(_stream(process_line),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# =====================================================================
#  Entry Point
# =====================================================================

if __name__ == '__main__':
    print("ESP32 Unified Vision API")
    print(f"Server starting on http://localhost:{PORT}")
    print(f"  /stream/raw    → Raw feed")
    print(f"  /stream/human  → Human detection")
    print(f"  /stream/signs  → Sign detection")
    print(f"  /stream/line   → Line follower")
    uvicorn.run(app, host='0.0.0.0', port=PORT)
