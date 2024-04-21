from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List
import base64
import cv2
import numpy as np

app = FastAPI()

class FaceData(BaseModel):
    """Data model for storing face recognition information including embeddings."""
    name: str
    timestamp: str
    frame: str  # Base64-encoded image of the face

@app.get("/")
def home():
    """Root endpoint for basic API check."""
    return {"message": "API is online"}

@app.post("/process_data", status_code=status.HTTP_201_CREATED)
async def process_data(face_data: FaceData):
    """
    Endpoint to process face recognition data received from another service, including embeddings.

    This endpoint accepts JSON payload containing the name, timestamp, and face embeddings of recognized individuals.
    It processes the information, which could involve logging it, storing it in a database, or using it to trigger further actions.

    Args:
        face_data: FaceData object that includes the name, timestamp, and embedding of the individual.

    Returns:
        A JSON response indicating that the data was received and processed.

    Raises:
        HTTPException: An error status code and message if an exception occurs.
    """
    try:
        # Decode the base64-encoded frame
        img_data = base64.b64decode(face_data.frame)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Log the frame shape
        if img is not None:
            print(f"Received frame shape: {img.shape}")  # img.shape returns a tuple (height, width, channels)
        else:
            print("Failed to decode image or image is corrupt.")
        # Log the received data
        print(f"Received data - Name: {face_data.name}, Timestamp: {face_data.timestamp}")

        # Respond that the data was processed successfully
        return {"message": f"Data processed - Name: {face_data.name}, Timestamp: {face_data.timestamp}"}

    except Exception as e:
        # Log the exception and return an HTTP error response
        print(f"Error processing data: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

