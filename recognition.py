import cv2
import numpy as np
import pyscreenshot as ImageGrab
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {}

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

X_COORD = 50
Y_COORD = 200
WIDTH = 800
HEIGHT = 350 + 200

while True:
    screen_grab = np.array(ImageGrab.grab(bbox= (X_COORD, Y_COORD, X_COORD + WIDTH, Y_COORD + HEIGHT)))
    gray = cv2.cvtColor(screen_grab,cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(screen_grab,cv2.COLOR_BGR2RGB)
    
    faces = detector.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30, 30),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(color, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w]) 
        print(id, confidence)

        if (confidence < 100):
            id = labels[id].capitalize()
            print(id)
        else:
            id = "Unknown"
        
        cv2.putText(
                    color, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )

    cv2.imshow('Real Time Face Recognition',color) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    
cv2.destroyAllWindows()


