import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

def get_encode(face_encoder, face, size):
    '''
    this method give embedding of face with FaceNet
    '''
    face = cv2.resize(face, size)
    encode = face_encoder.embeddings(np.expand_dims(face, axis=0))[0]
    return encode


def get_face(img, box):
    '''
    this method give us the actual value of box and then return face part in image and values of location
    '''
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


l2_normalizer = Normalizer('l2')


def plt_show(cv_img):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def load_pickle(path):
    '''
    load pickle file
    '''
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def save_pickle(path, obj):
    '''
    save data in pickle format 
    '''
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
