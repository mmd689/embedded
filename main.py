import cv2
import face_recognition
import time
from os import walk

# cap = cv2.VideoCapture('http://192.168.75.16:4747/mjpegfeed?640x480')
cap = cv2.VideoCapture(0)

previously_added_faces = []
previously_added_names = []

f = []
for (dirpath, dirnames, filenames) in walk("pictures"):
    for filename in filenames:
        im = cv2.imread("pictures/" + filename, cv2.IMREAD_COLOR)
        face_locations = face_recognition.face_locations(im)
        # for (top, right, bottom, left) in face_locations:
        #     im = im[top:bottom, left: right]
        #     break
        cv2.imshow("image", im)
        cv2.waitKey(0)
        previously_added_faces.append(face_recognition.face_encodings(im)[0])
        s_1 = str.split(filename)
        s_2 = str.split(s_1[2], '.')
        previously_added_names.append((s_1[0], s_1[1], s_2[0]))
    break



time_ = time.time()
face_locations = []
while True:
    ret, frame = cap.read()
    if time.time() - time_ > 2:
        time_ = time.time()
    # face_locations = face_recognition.face_locations(frame)
    # print(face_locations)
    # for (top, right, bottom, left) in face_locations:
    #     im = frame[top:bottom, left: right]
    im_encode = face_recognition.face_encodings(frame)[0]
    result = face_recognition.compare_faces(previously_added_faces, im_encode)
    print(result)
