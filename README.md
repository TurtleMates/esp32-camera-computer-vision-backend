# ESP32 Camera — Computer Vision Backend

A Python backend that connects to an **ESP32-CAM** module over Wi-Fi and serves multiple real-time computer vision streams through a FastAPI web server. It provides raw video passthrough, YOLOv8-based human detection & segmentation, road sign detection, and a black-tape line-following algorithm — all from a single unified API.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Standalone Scripts](#standalone-scripts)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

| Stream | Endpoint | Description |
|--------|----------|-------------|
| **Raw Feed** | `/stream/raw` | Unprocessed camera feed from the ESP32 |
| **Human Detection** | `/stream/human` | Real-time human detection & instance segmentation (YOLOv8) |
| **Road Signs** | `/stream/signs` | Stop sign & traffic light detection (YOLOv8) |
| **Line Follower** | `/stream/line` | Black tape line detection with steering guidance (OpenCV) |

### Key Design Decisions

- **Single camera connection** — A shared `CameraReader` thread maintains one connection to the ESP32, so all streams share frames without overwhelming the camera.
- **Single YOLO model** — One `yolov8n-seg` model is loaded once and shared between human and sign detection to minimize memory usage.
- **CUDA warmup** — A dummy inference runs at startup to pre-compile CUDA kernels, eliminating the 2–5 s cold-start delay on the first request.

---

## Architecture

```
┌──────────────┐       Wi-Fi / HTTP        ┌──────────────────────────┐
│  ESP32-CAM   │  ───────────────────────►  │  CameraReader (thread)   │
│  /stream     │    MJPEG stream            │  Reads frames into a     │
└──────────────┘                            │  shared buffer           │
                                            └──────────┬───────────────┘
                                                       │  latest frame
                              ┌─────────────┬──────────┼──────────┬─────────────┐
                              ▼             ▼          ▼          ▼             │
                         process_raw   process_human process_signs process_line │
                          (passthru)    (YOLOv8)     (YOLOv8)    (OpenCV)      │
                              │             │          │          │             │
                              ▼             ▼          ▼          ▼             │
                         /stream/raw  /stream/human /stream/signs /stream/line  │
                              │             │          │          │             │
                              └─────────────┴──────────┴──────────┘             │
                                            │                                  │
                                     FastAPI Server                            │
                                     (port 5000)                               │
                                            │                                  │
                                     Browser / Client ◄────────────────────────┘
```

---

## Project Structure

```
esp32/
├── app.py                    # ★ Main unified server (recommended entry point)
├── video_stream.py           # Standalone: raw feed viewer (OpenCV window)
├── yolo_stream.py            # Standalone: human detection server (port 5000)
├── yolo_signs_stream.py      # Standalone: sign detection server (port 5001)
├── lane_detection_stream.py  # Standalone: line follower server (port 5002)
├── yolov8n-seg.pt            # YOLOv8 nano segmentation weights (auto-downloaded)
├── venv/                     # Python virtual environment
├── .gitignore
└── README.md                 # ← You are here
```

### File Roles

| File | Purpose |
|------|---------|
| `app.py` | **Unified API** — runs all 4 streams from a single FastAPI server on port 5000. This is the main file you should use. |
| `video_stream.py` | Simple OpenCV script that opens the ESP32 stream in a desktop window. Useful for quick debugging. |
| `yolo_stream.py` | Standalone FastAPI server for human detection only (port 5000). Early prototype, superseded by `app.py`. |
| `yolo_signs_stream.py` | Standalone FastAPI server for road sign detection only (port 5001). Early prototype, superseded by `app.py`. |
| `lane_detection_stream.py` | Standalone FastAPI server for line following only (port 5002). Early prototype, superseded by `app.py`. |

---

## Prerequisites

- **Python 3.9+**
- **ESP32-CAM** module flashed with a firmware that serves an MJPEG stream at `http://<ESP32_IP>/stream`
- **NVIDIA GPU** (optional but recommended) — YOLO runs on CPU if no GPU is available, but inference will be slower
- **pip** package manager

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TurtleMates/esp32-camera-computer-vision-backend.git
cd esp32-camera-computer-vision-backend
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn opencv-python numpy ultralytics
```

> **Note:** The first run will automatically download the `yolov8n-seg.pt` model weights (~7 MB) if they are not already present.

### 4. (Optional) GPU support

If you have an NVIDIA GPU and want accelerated inference:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Configuration

All configuration is done through constants at the top of `app.py`. Edit these before starting the server:

### Network

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_URL` | `http://192.168.137.159/stream` | URL of the ESP32-CAM MJPEG stream. **Change this to your ESP32's IP address.** |
| `PORT` | `5000` | Port the FastAPI server listens on |

### YOLO

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n-seg.pt` | YOLOv8 model weights file |
| `HUMAN_CONF` | `0.5` | Minimum confidence for human detection |
| `SIGNS_CONF` | `0.4` | Minimum confidence for sign detection |

### Line Follower

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAHE_CLIP` | `3.0` | CLAHE contrast limit (higher = more enhancement) |
| `CLAHE_TILE` | `(8, 8)` | CLAHE tile grid size |
| `MEDIAN_K` | `7` | Median blur kernel size |
| `GAUSS_K` | `(7, 7)` | Gaussian blur kernel size |
| `ERODE_K` / `ERODE_ITER` | `5` / `2` | Erosion parameters for noise removal |
| `DILATE_K` / `DILATE_ITER` | `7` / `2` | Dilation parameters |
| `CLOSE_K` | `15` | Morphological closing kernel |
| `OPEN_K` | `11` | Morphological opening kernel |
| `MIN_CONTOUR_AREA` | `2000` | Minimum contour area (px²) to be considered tape |
| `MIN_STRIP_PX` | `10` | Minimum white pixels in a horizontal slice |
| `NUM_SLICES` | `15` | Number of horizontal slices for path tracking |
| `CENTER_DEADZONE` | `25` | Pixel range from center considered "straight" |

---

## Running the Server

### 1. Update the ESP32 IP

Open `app.py` and set `STREAM_URL` to your ESP32-CAM's IP address:

```python
STREAM_URL = "http://<YOUR_ESP32_IP>/stream"
```

### 2. Start the server

```bash
python app.py
```

You should see:

```
Warming up YOLO model...
Model ready!
ESP32 Unified Vision API
Server starting on http://localhost:5000
  /stream/raw    → Raw feed
  /stream/human  → Human detection
  /stream/signs  → Sign detection
  /stream/line   → Line follower
```

### 3. Open in a browser

Navigate to any of these URLs:

| URL | Stream |
|-----|--------|
| `http://localhost:5000/` | Landing page with links to all streams |
| `http://localhost:5000/stream/raw` | Raw camera feed |
| `http://localhost:5000/stream/human` | Human detection & segmentation |
| `http://localhost:5000/stream/signs` | Stop sign & traffic light detection |
| `http://localhost:5000/stream/line` | Line follower with steering arrows |
| `http://localhost:5000/docs` | Interactive Swagger API documentation |

---

## API Endpoints

### `GET /`

Returns an HTML landing page listing all available streams.

### `GET /stream/raw`

Returns the raw ESP32 camera feed as an MJPEG stream.

- **Content-Type:** `multipart/x-mixed-replace; boundary=frame`
- **Processing:** None (passthrough)

### `GET /stream/human`

Returns the camera feed with YOLOv8 human detection overlaid.

- **Content-Type:** `multipart/x-mixed-replace; boundary=frame`
- **Model:** YOLOv8n-seg (COCO class 0 = person)
- **Confidence threshold:** 0.5

### `GET /stream/signs`

Returns the camera feed with road sign detection overlaid.

- **Content-Type:** `multipart/x-mixed-replace; boundary=frame`
- **Model:** YOLOv8n-seg (COCO class 9 = traffic light, class 11 = stop sign)
- **Confidence threshold:** 0.4

### `GET /stream/line`

Returns a side-by-side view: annotated frame (left) and binary mask (right).

- **Content-Type:** `multipart/x-mixed-replace; boundary=frame`
- **Output:** Shows tape contour (green), path centroids (yellow), path polyline (magenta), and a steering arrow with direction label (CENTER / LEFT / RIGHT)

---

## Standalone Scripts

These scripts were developed as prototypes before being consolidated into `app.py`. They can still be useful for quick testing or running a single vision task.

### `video_stream.py` — Raw Feed Viewer

Opens the ESP32 stream in a local OpenCV window. Press **`q`** to quit.

```bash
python video_stream.py
```

> **Note:** Edit the `stream_url` variable at the top of the file to match your ESP32 IP.

### `yolo_stream.py` — Human Detection (Standalone)

Runs a FastAPI server on **port 5000** with a single `/stream` endpoint for human detection.

```bash
python yolo_stream.py
# → http://localhost:5000/stream
```

### `yolo_signs_stream.py` — Sign Detection (Standalone)

Runs a FastAPI server on **port 5001** with a single `/stream` endpoint for road sign detection.

```bash
python yolo_signs_stream.py
# → http://localhost:5001/stream
```

### `lane_detection_stream.py` — Line Follower (Standalone)

Runs a FastAPI server on **port 5002** with a single `/stream` endpoint for black tape line following.

```bash
python lane_detection_stream.py
# → http://localhost:5002/stream
```

---

## How It Works

### Camera Reader

The `CameraReader` class (in `app.py`) opens a single connection to the ESP32-CAM's MJPEG stream in a background thread. It continuously reads frames into a shared buffer protected by a threading lock. All stream endpoints read from this same buffer, so the camera is never overwhelmed by multiple concurrent connections.

If the camera connection drops, it automatically reconnects after a 1-second delay.

### Human Detection (`/stream/human`)

Uses **YOLOv8n-seg** (a nano-sized segmentation model trained on COCO). It filters for class `0` (person) and overlays bounding boxes + instance segmentation masks on the frame.

### Road Sign Detection (`/stream/signs`)

Uses the same YOLOv8 model but filters for classes `9` (traffic light) and `11` (stop sign). Overlays detections on the frame.

### Line Follower (`/stream/line`)

A multi-stage OpenCV pipeline designed for detecting a black tape line on a floor, optimized for the ESP32-CAM's typically dark and noisy images:

1. **CLAHE Enhancement** — Adaptive histogram equalization to boost contrast in dark areas
2. **Blur** — Median + Gaussian blur to reduce sensor noise
3. **Otsu Thresholding** — Automatic binary threshold that separates the dark tape from the floor
4. **Morphological Cleanup** — Erosion → dilation → closing → opening to remove small noise blobs and fill gaps in the tape
5. **Contour Detection** — Finds the largest contour (the tape) and validates it by area
6. **Path Tracking** — Divides the frame into 15 horizontal slices and computes the centroid of tape pixels in each slice to build a path
7. **Steering Guidance** — Compares the bottom-most centroid to the frame center and draws a directional arrow (LEFT / CENTER / RIGHT) with pixel deviation

The output is a side-by-side frame: the annotated image on the left and the binary mask on the right, making it easy to debug thresholding issues.

### MJPEG Streaming

All endpoints stream frames as **MJPEG** (Motion JPEG) using the `multipart/x-mixed-replace` HTTP content type. This is natively supported by all modern browsers — just point an `<img>` tag or open the URL directly. The stream runs at approximately **20 FPS** (throttled via a 50 ms sleep per frame).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Error: Could not open stream` | Verify the ESP32 is powered on and connected to Wi-Fi. Check that `STREAM_URL` matches the ESP32's IP. Try opening `http://<ESP32_IP>/stream` in a browser first. |
| YOLO is very slow | You're likely running on CPU. Install PyTorch with CUDA support (see [Installation](#installation)). |
| First request is slow | This is normal on first startup — the YOLO warmup should handle it. If you still see delays, ensure the warmup completed ("Model ready!" in console). |
| Line follower not detecting tape | Adjust `CLAHE_CLIP` (increase for darker environments), `MIN_CONTOUR_AREA` (decrease for thin tape), or the morphology parameters. |
| Stream is laggy / low FPS | The ESP32-CAM has limited bandwidth. Reduce the camera resolution in your ESP32 firmware, or ensure you're on a strong Wi-Fi signal. |
| Port already in use | Another process is using port 5000. Change `PORT` in `app.py` or kill the existing process. |
| CORS errors from a frontend | CORS is already enabled for all origins (`*`). If you still see errors, check that you're connecting to the right port. |

---

## License

This project is open source. Feel free to use, modify, and distribute.
