import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from utils import save_pickle, l2_normalizer, get_face

# Face Detector
detector = MTCNN()

# Face Recognizer
Myfacenet = FaceNet()

# change size of face for pass to facenet NN
required_Size = (160, 160)

# dict for save embbeddings face of peoples
encoding_dict = dict()

# directory of person images
people_dir = 'person'

# encoding directory
encodings_path = 'encodings.pkl'


for person_name in os.listdir(people_dir):
    person_dir = os.path.join(people_dir, person_name)
    encodes = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detct faces in image
        results = detector.detect_faces(img_rgb)
        if results:

            # get max for face in image with high confidence
            res = max(results, key=lambda b: b['box'][2] * b['box'][3])
            face, _, _ = get_face(img_rgb, res['box'])
            face = cv2.resize(face, required_Size)
            
            # recognize face and extrcat embeddings
            encode = Myfacenet.embeddings(np.expand_dims(face, axis=0))[0]
            encodes.append(encode)
    if encodes:

        # use np.sum for peoples that have more than 1 image (we can use np.mean also)
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[person_name] = encode

# save encodings of person faces
save_pickle(encodings_path, encoding_dict)
