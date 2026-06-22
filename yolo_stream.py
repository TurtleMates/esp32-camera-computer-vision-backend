import cv2
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

# URL of the original video stream
stream_url = "http://10.200.7.27/stream"

# Load the YOLOv8 segmentation model
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

        # Run YOLOv8 segmentation on the frame (classes=0 for 'person')
        results = model(frame, classes=0, conf=0.5)

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

    cap.release()

@app.get('/stream')
def video_feed():
    """Video streaming route for FastAPI."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    print("Starting the FastAPI server...")
    print("Access the processed stream at: http://localhost:5000/stream")
    # Run the FastAPI app using uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
