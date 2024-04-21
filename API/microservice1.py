from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import shutil
from io import BytesIO
import pytz
import datetime
import requests

app = FastAPI()
face_detection_url = "http://localhost:8001/detect_faces"

@app.get("/")
def home():
    """Root endpoint to check service health."""
    return {"message": "Service is online."}

@app.post("/upload/")
async def read_video(file: UploadFile = File(...)):
    """Endpoint to upload and process video files for face detection.

    Args:
        file: A video file uploaded by the client.

    Returns:
        A JSON response indicating the outcome of the operation.
    """
    # Validate the file format
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        return JSONResponse(content="Invalid file format", status_code=400)

    # Initialize variables
    frame_index = 0
    files = []
    batch_size = 500  # Number of frames per batch for processing

    # Process the video
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, 'wb') as out_file:
            shutil.copyfileobj(file.file, out_file)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return JSONResponse(content="Unable to load the video", status_code=400)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Timestamp the frame in the Asia/Tehran timezone
            eastern = pytz.timezone('Asia/Tehran')
            now = datetime.datetime.now(eastern)
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds

            # Convert frame to JPEG and prepare for sending
            _, buffer = cv2.imencode('.jpg', frame)
            buffer = BytesIO(buffer.tobytes())
            buffer.seek(0)
            files.append(('frames', (f"frame_{len(files)}.jpg", buffer, 'image/jpeg')))
            files.append(('timestamps', (f"timestamp_{frame_index}.txt", timestamp, 'text/plain')))
            frame_index += 1

            # Send batch if limit is reached
            if len(files) >= batch_size * 2:
                response = requests.post(face_detection_url, files=files)
                if response.status_code != 200:
                    cap.release()
                    return JSONResponse(content=f"Error sending frames: {response.text}", status_code=500)
                files = []

        # Clean up and send any remaining frames
        cap.release()
        if files:
            response = requests.post(face_detection_url, files=files)
            if response.status_code != 200:
                return JSONResponse(content=f"Error sending frames: {response.text}", status_code=500)

        return JSONResponse(content="Great! Frames processed and sent.", status_code=200)

    except Exception as e:
        return JSONResponse(content=str(e), status_code=500)

