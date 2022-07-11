import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Enter user ID end press <Enter> ==>  ')
print("\n Initializing face capture. Look the camera and wait ...")

count = 0
while True:
    ret, img = cam.read()
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", img[y:y + h, x:x + w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

cam.release()
cv2.destroyAllWindows()
