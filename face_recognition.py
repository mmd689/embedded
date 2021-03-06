import cv2
import numpy as np
import os
from check_glasses import check_glasses

model = cv2.face.LBPHFaceRecognizer_create()
model.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
person_id = 0

names = ['None', 'Bahar', 'Dorna', "MohammadAmin"]

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("current-frame.png", gray)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        person_id, confidence = model.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if confidence < 100:
            person_id = names[person_id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            person_id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(person_id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    print("Does {} has glasses? {}".format(person_id, str(check_glasses("current-frame.png"))))

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
