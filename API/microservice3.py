from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict
from keras_facenet import FaceNet
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import cv2
import requests
from pydantic import BaseModel
import base64

# Load the FaceNet model and encoding dictionary
def load_pickle(path: str) -> Dict:
    """Load a pickle file from the given path."""
    with open(path, 'rb') as file:
        return pickle.load(file)

encoding_dict = load_pickle('../encodings.pkl')
Myfacenet = FaceNet()
app = FastAPI()
class MatchData(BaseModel):
    name: str
    timestamp: str
    embedding: list[float]

@app.get("/")
def home():
    """Root endpoint for basic API check."""
    return {"message": "API is online"}

@app.post("/process_faces/")
async def process_faces(frames: list[UploadFile] = File(...), timestamps: list[UploadFile] = File(...)):
    """
    Endpoint that receives detected face images and their timestamps,
    performs face recognition, and optionally forwards match data to another service.
    """
    try:
        for frame_file, timestamp_file in zip(frames, timestamps):
            # Read the timestamp
            timestamp = await timestamp_file.read()
            timestamp = timestamp.decode('utf-8').strip()

            # Read and decode the frame
            contents = await frame_file.read()
            np_img = np.frombuffer(contents, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (160, 160))

            # Calculate the embedding for the face
            embedding = Myfacenet.embeddings(np.expand_dims(frame, axis=0))[0]
            name = 'unknown'
            min_distance = float("inf")

            # Determine if the face matches any in the encoding dictionary
            for db_name, db_embedding in encoding_dict.items():
                dist = cosine(db_embedding, embedding)
                if dist < 0.3 and dist < min_distance:
                    name = db_name
                    min_distance = dist

            # If a match is found, send data to the fourth service
            if name != 'unknown':
                embedding_list = embedding.tolist()
                # Encode the frame to JPEG and then to base64
                retval, buffer = cv2.imencode('.jpg', frame)
                if retval:
                    jpg_as_text = base64.b64encode(buffer).decode()
                else:
                    raise ValueError("Could not encode image to JPEG")

                data = {
                    "name": name,
                    "timestamp": timestamp,
                    "frame": jpg_as_text
                }
                response = requests.post("http://localhost:8005/process_data", json=data)
                print(f"Match found: {name} at {timestamp}")
                print(f"Response from fourth service: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing frame: {str(e)}")

