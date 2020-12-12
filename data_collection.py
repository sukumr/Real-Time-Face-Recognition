import numpy as np
import cv2
import pyscreenshot as ImageGrab
import os

# face detection model
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# bounding box parameters
X_COORD = 50
Y_COORD = 200
WIDTH = 500 + 350
HEIGHT = 600

img_name = input("Please enter the name of a image:")

current_dir = 'dataset'

new_dir = os.path.join(current_dir, img_name)

if not os.path.exists(new_dir):
    os.makedirs(new_dir)
    count = 1
else:
    count = len(os.listdir(new_dir)) + 1

while True:
    screen_grab = np.array(ImageGrab.grab(bbox= (X_COORD, Y_COORD, X_COORD + WIDTH, Y_COORD + HEIGHT)))
    gray = cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB)
    color = cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(
        gray,     # input greyscale image
        scaleFactor=1.2,
        minNeighbors=5,  
        minSize=(20, 20) # min rect covering face
    )

    for (x,y,w,h) in faces: # x, y, w, h: region of interest 
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2) # color = (BGR)
        
    if cv2.waitKey(50) & 0xFF == ord("c"):
        img_id = f'{img_name}{count}'
        count += 1
        cv2.imwrite(f"{new_dir}/{img_id}.png", gray[y : y + h, x : x + w])
        print("Captured:", img_id)

    cv2.imshow("frame", gray)
    k = cv2.waitKey(30) & 0xff # Press 'ESC' for exiting
    if k == 27:
        break

cv2.destroyAllWindows()
