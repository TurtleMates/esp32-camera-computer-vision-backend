import cv2

# URL of the video stream
stream_url = "http://10.200.7.27/stream"

# Create a VideoCapture object
cap = cv2.VideoCapture(stream_url)

# Check if the stream was successfully opened
if not cap.isOpened():
    print("Error: Could not open the video stream.")
    exit()

print("Successfully connected to the stream. Press 'q' to quit.")

while True:
    # Read a frame from the stream
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to receive frame from the stream. Exiting ...")
        break

    # --- Add your edits and modifications here ---
    # Example: Convert the frame to grayscale
    # modified_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    modified_frame = frame  # Passing the frame as is for now
    # ---------------------------------------------

    # Display the resulting frame
    cv2.imshow("ESP32 Video Stream", modified_frame)

    # Press 'q' on the keyboard to exit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
