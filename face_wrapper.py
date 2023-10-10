from mtcnn import MTCNN
import cv2
import os
import urllib.request
import numpy as np

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

from PIL import Image
from tqdm import tqdm

class FaceWrapper(object):
    def __init__(self):
        self.mtcnn_detector = MTCNN()
        
        self.haarcascade_url = "https://raw.githubusercontent.com/anaustinbeing/\
                                        haar-cascade-files/master/haarcascade_frontalface_default.xml"
        self.download_haarcascade()

        self.hc_detector = cv2.CascadeClassifier(
            os.path.basename(self.haarcascade_url))
    
    def download_haarcascade(self):
        if os.path.exists(
            os.path.basename(self.haarcascade_url)): return True
        
        urllib.request.urlretrieve(self.haarcascade_url, 
                                   os.path.basename(self.haarcascade_url))

    def get_mtcnn_faces(self, image):
        results = self.mtcnn_detector.detect_faces(image)
        for result in results:
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height

            face = image[y1:y2, x1:x2]

            yield face

    def get_hc_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = self.hc_detector.detectMultiScale(gray, scaleFactor=1.05,
                        minNeighbors=5, minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in rects:
            yield [x, y, w, h]
    

class FaceProcessor(object):
    def __init__(self, filepaths: str):
        self.filepaths = filepaths
        self.faces = [Image.open(path) for path in tqdm(filepaths)]
        self.embeddings = None
        self.model = VGGFace(
            model='resnet50', 
            include_top=False, 
            input_shape=(224, 224, 3), 
            pooling='avg')
    
    def filter(self, height: int = 60, width: int = 60):
        
        new_filepaths = []
        new_faces = []
        for filepath, face in tqdm(zip(self.filepaths, self.faces)):
            h, w = face.size
            if h < height or w < width:
                continue
            new_faces.append(face)
            new_filepaths.append(filepath)
                
        self.filepaths = new_filepaths
        self.faces = new_faces
    
    def calculate_embeddings(self):
        np_faces = [np.asarray(face.resize((224, 224))) for face in tqdm(self.faces)]
        np_faces = [preprocess_input(face.astype(np.float32), version=2) for face in tqdm(np_faces)]
        np_faces = np.asarray(np_faces)
        self.embeddings = self.model.predict(np_faces)
    
    def fetch_embedding(self, filepath: str):
        image = np.asarray(Image.open(filepath).resize((224, 224)))
        image = np.expand_dims(preprocess_input(image.astype(np.float32), version=2), axis=0)
        return self.model(image, training=False)

    def __len__(self):
        return len(self.faces)
    
    def __getitem__(self, idx):
        return self.faces[idx]

# from glob import glob
# filenames = glob("images/FACE-*.jpeg")
# face_processor = FaceProcessor(filenames)
# print(len(face_processor))

# face_processor.filter(60, 60)
# print(len(face_processor))
# print(face_processor.calculate_embeddings())
# print(face_processor.embeddings.shape)