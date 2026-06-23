import cv2
import time
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

# URL of the original video stream
stream_url = "http://10.200.7.27/stream"

# Load the YOLOv8 segmentation model
# Note: The standard YOLO model natively recognizes 'stop sign' and 'traffic light'
model = YOLO('yolov8n-seg.pt')

def generate_frames():
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: Could not open the video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLOv8 segmentation on the frame
        # In the COCO dataset:
        # class 9 = traffic light
        # class 11 = stop sign
        results = model(frame, classes=[9, 11], conf=0.4)

        # Plot the results on the frame
        annotated_frame = results[0].plot()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        # Yield the output frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Throttle to reduce lag (~10 FPS)
        time.sleep(0.1)

    cap.release()

@app.get('/stream')
def video_feed():
    """Video streaming route for FastAPI."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    print("Starting the FastAPI server for Road Signs...")
    print("Access the processed stream at: http://localhost:5001/stream")
    # Running on port 5001 so it doesn't conflict if the other one is running
    uvicorn.run(app, host='0.0.0.0', port=5001)
