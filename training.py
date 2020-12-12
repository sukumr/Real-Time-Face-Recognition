import os
import cv2
import numpy as np
import pickle

from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create() # LBPH: Local binary pattern histogram
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

dataDir = 'dataset'

current_id = 0
label_ids = {}
faceImages = []
faceLabels = []

for root, dirs, files in os.walk(dataDir):
    for f in files:
        path = os.path.join(root, f)
        label = os.path.basename(root).replace(" ", "-").lower() # replace space with dash
        # print(label, path)

        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        lab_id = label_ids[label]
        # print(label_ids)

        pil_img = Image.open(path).convert('L')
        img_numpy = np.array(pil_img,'uint8')
        # print(img_numpy)
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
                faceImages.append(img_numpy[y:y+h,x:x+w]) # only face is extracted and used for training
                faceLabels.append(lab_id)

print(label_ids)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f) # dumping label_ids in to labels.pickle file

recognizer.train(faceImages, np.array(faceLabels))
recognizer.write('trainer.yml') 