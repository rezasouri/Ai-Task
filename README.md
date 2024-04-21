# Face Detection and Log Data

## Description
This project utilizes a microservices architecture to detect and recognize faces from video streams. The workflow is divided into four stages, each handled by a separate microservice. Here’s how it works:

- **First Microservice**: Receives video input and preprocesses the frames.
- **Second Microservice**: Uses MTCNN to detect faces in each frame and forwards the detected faces along with timestamps to the third microservice.
- **Third Microservice**: Employs FaceNet to recognize faces against a pre-existing database. If a face is recognized, it sends the face data, the recognized person's name, and the timestamp to the fourth microservice.
- **Fourth Microservice**: Logs the recognized face data along with the name and timestamp.

All microservices are implemented using FastAPI and communicate with each other via POST requests.

## Installation

Before running the microservices, ensure that Docker and Docker Compose are installed on your system. Follow these steps to set up the project:

1. Clone the repository:

git clone https://github.com/your-username/face-detection-and-log-data.git
cd face-detection-and-log-data

2. Build and run the containers:

docker-compose up --build


This command builds the images if they don’t exist and starts the containers.

## Usage

Once the services are running, you can send a video stream to the first microservice's endpoint to begin the face detection and recognition process. Use the following endpoint to post video data:

POST /api/v1/process-video
Content-Type: multipart/form-data


Attach the video file with the key `video`.

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Contact

For any further queries, please contact [Your Name] at [your-email@example.com].

## Additional Data

For those who prefer to run face detection locally without using the API, you can utilize the scripts provided in the `Model` directory. Here's how to use them:

### 1. **Encoding Images**
   - Navigate to the `Model` directory.
   - Run the `encoding.py` script to create embedding data from your images. This step preprocesses your images and creates a dataset of face embeddings that can be used for recognition.

### 2. **Recognizing Faces in Images**
   - Use the `face_detection_images.py` script to recognize faces within static images. This script will analyze each image, detect faces, and attempt to recognize them based on the embeddings created previously.

### 3. **Recognizing Faces in Videos**
   - For real-time face recognition, run the `face_detection_video.py` script. This script is capable of recognizing faces in video streams, including live feeds from a webcam. It processes video frames in real time, detects and recognizes faces, and displays the results.

These tools provide a flexible way to implement face recognition technology directly on your local system, either for development or testing purposes. Feel free to explore and modify the scripts according to your needs!

