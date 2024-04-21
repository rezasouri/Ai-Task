<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://miro.medium.com/v2/resize:fit:1400/1*MNj7uq7HUNGERaYgRRdZfw.jpeg" alt="Project logo"></a>
</p>

<h3 align="center">Face Detection and Log Data</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Face Recognition Microservices With FastApi and AI Tools
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Aditional Data](#additional)

## 🧐 About <a name = "about"></a>

This project utilizes a microservices architecture to detect and recognize faces from video streams. The workflow is divided into four stages, each handled by a separate microservice. Here’s how it works:

- **First Microservice**: Receives video input and preprocesses the frames.
- **Second Microservice**: Uses MTCNN to detect faces in each frame and forwards the detected faces along with timestamps to the third microservice.
- **Third Microservice**: Employs FaceNet to recognize faces against a pre-existing database. If a face is recognized, it sends the face data, the recognized person's name, and the timestamp to the fourth microservice.
- **Fourth Microservice**: Logs the recognized face data along with the name and timestamp.

All microservices are implemented using FastAPI and communicate with each other via POST requests.

## 🏁 Getting Started <a name = "getting_started"></a> and Installation

Before running the microservices, ensure that Docker and Docker Compose are installed on your system.For [Docker](https://www.docker.com/) Instalation see link. Follow these steps to set up the project:

### Prerequisites

Clone the repository:

```
git clone https://github.com/rezasouri/Ai-Task.git
cd AI-Task
```
### Using Docker

First Navigate to `Docker` directory

```
cd Docker
```


Build and run the containers:

```
docker-compose up --build
```
This command builds the images if they don’t exist and starts the containers.

### Running Locally Without Docker

If you prefer not to use Docker, you can run the APIs locally:

1. Navigate to the `API` directory.
2. Run each service using Python in `Terminal` with `Uvicorn`. For example:


``` bash
uvicorn script_name:app --port PORT_NUMBER --host HOST --reload
```

- `script_name`: Replace this with the name of your Python file that contains the FastAPI app instance (e.g., `main.py`).
- `app`: This should be the name of the FastAPI instance in your script.
- `PORT_NUMBER`: Specify the port number on which the server should listen (e.g., `8000`).
- `HOST`: Define the host address (e.g., `127.0.0.1` for local development or `0.0.0.0` to allow external access).
- `--reload`: This optional flag enables auto-reload on code changes, which is very useful during development.



## 🎈 Usage <a name="usage"></a>

Once the services are running, you can send a video stream to the first microservice's endpoint to begin the face detection and recognition process. Use the following endpoint to post video data:
```
curl -X POST http://localhost:8000/docs#/default/read_video_upload__post \
     -F "file=@/path/to/your/video.file" \
     -H "Content-Type: multipart/form-data"


```

Or you can go to 

```
http:\\localhost:8000/docs
```

and send post request (select video and send it)

![Alt text](/assets/image.png "Optional title")

### In Docker
You can use this command for see logs data

```
docker logs [container id]
```

### Results 
Results are look like below:

![Alt text](/assets/results.png "Optional title")


## ⛏️ Built Using <a name = "built_using"></a>

- [FastApi](https://fastapi.tiangolo.com/) - Web Framework
- [Docker](https://www.docker.com/) - Containerization Platform


## ✍️ Authors <a name = "authors"></a>

- [@rezasouri](https://github.com/rezasouri) - Idea & Initial work



## 📦 Additional Data <a name = "additional"></a>

For those who prefer to run face detection locally without using the API, you can utilize the scripts provided in the `Model` directory. Here's how to use them:

### 1. **Encoding Images**
   - Navigate to the `Model` directory.
   - Run the `encoding.py` script to create embedding data from your images. This step preprocesses your images and creates a dataset of face embeddings that can be used for recognition.

### 2. **Recognizing Faces in Images**
   - Use the `face_detection_images.py` script to recognize faces within static images. This script will analyze each image, detect faces, and attempt to recognize them based on the embeddings created previously.

### 3. **Recognizing Faces in Videos**
   - For real-time face recognition, run the `face_detection_video.py` script. This script is capable of recognizing faces in video streams, including live feeds from a webcam. It processes video frames in real time, detects and recognizes faces, and displays the results.

These tools provide a flexible way to implement face recognition technology directly on your local system, either for development or testing purposes. Feel free to explore and modify the scripts according to your needs!
