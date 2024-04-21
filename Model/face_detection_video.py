import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from utils import  l2_normalizer, get_face, get_encode, load_pickle

# encoding path
encoding_path = 'encodings.pkl'

# Face Detector
detector = MTCNN()

# Face Recognizer
Myfacenet = FaceNet()

# change size of face for pass to facenet NN
required_Size = (160, 160)

# load encoding from pickle file
encoding_dict = load_pickle(encoding_path)

def video_camera(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(Myfacenet, face, required_Size)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

        name = 'unknown'
        distance = float("inf")

        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < 0.3 and dist < distance:
                name = db_name
                distance = dist
        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', pt_1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return img


if __name__ == '__main__':
    vc = cv2.VideoCapture(0)
    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print("no frame:()")
            break
        frame = video_camera(frame)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break