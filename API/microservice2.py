from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import base64
import logging
import shutil
from io import BytesIO
import requests  # Ensure only one requests import is used

# Constants
DATA_PROCESSING_URL = "http://localhost:8003/process_faces"
BATCH_SIZE = 500  # Maximum number of face images per batch

# Initialize FastAPI and MTCNN detector
app = FastAPI()
detector = MTCNN()  # Load MTCNN model for face detection

@app.get("/")
def home():
    """Root endpoint to check service health."""
    return {"message": "Service is online."}

@app.post("/detect_faces/")
async def detect_faces(frames: list[UploadFile] = File(...), timestamps: list[UploadFile] = File(...)):
    """Endpoint to receive video frames and timestamps, detect faces, and send them for processing.

    Args:
        frames: A list of video frames as uploaded files.
        timestamps: A list of timestamps corresponding to each frame.

    Returns:
        A JSON response with the result of the operation.
    """
    frames_data = []  # Container for the data to be sent in batches

    try:
        for frame_index, (frame_file, timestamp_file) in enumerate(zip(frames, timestamps)):
            # Process the timestamp
            timestamp = await timestamp_file.read()
            timestamp = timestamp.decode('utf-8').strip()
            print(f"Timestamp for frame {frame_index}: {timestamp}")

            # Read and decode the frame
            frame_contents = await frame_file.read()
            frame_np_array = np.frombuffer(frame_contents, dtype=np.uint8)
            frame_image = cv2.imdecode(frame_np_array, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            detected_faces = detector.detect_faces(frame_rgb)

            for face_count, face_data in enumerate(detected_faces):
                face, top_left, bottom_right = get_face(frame_rgb, face_data['box'])

                # Convert face to JPEG format
                _, buffer = cv2.imencode('.jpg', face)
                buffer = BytesIO(buffer.tobytes())
                buffer.seek(0)

                # Prepare files for sending
                files = [
                    ('frames', (f"frame_{frame_index}_face_{face_count}.jpg", buffer, 'image/jpeg')),
                    ('timestamps', (f"timestamp_{frame_index}.txt", timestamp, 'text/plain'))
                ]
                frames_data.extend(files)

                # Check and send the batch if full
                if len(frames_data) // 2 >= BATCH_SIZE:
                    response = requests.post(DATA_PROCESSING_URL, files=frames_data)
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                    frames_data = []

        # Process any remaining frames
        if frames_data:
            response = requests.post(DATA_PROCESSING_URL, files=frames_data)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

        return JSONResponse(content={"message": "Data successfully processed and sent.", "response": response.json()})

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error when sending data to the service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")

    except Exception as e:
        logging.error(f"Error processing the frames: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing frame: {str(e)}")

def get_face(img, box):
    """Extracts a face from the image using the coordinates in the box.

    Args:
        img: The image from which to extract the face.
        box: The coordinates (x, y, width, height) of the face.

    Returns:
        The face image and the top-left and bottom-right points of the face.
    """
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)
