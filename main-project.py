import cv2
from check_glasses import check_glasses
from check_distance import get_distance

model = cv2.face.LBPHFaceRecognizer_create()
model.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
person_id = 0

names = ['None', 'Bahar', 'Dorna', "MohammadAmin"]
persons = ["Bahar", "Dorna", "Mohammadamin"]
has_glasses = {
    "Bahar": False,
    "Dorna": True,
    "Mohammadamin": True
}

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

unknown_person_alert = False
short_distance_alert = False
no_glasses_alert = False

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("current-frame.png", gray)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
    )

    for (x, y, w, h) in faces:
        person_id, confidence = model.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            person_id = names[person_id]
        else:
            person_id = "unknown"

    cv2.imshow('camera', img)
    img_has_glasses = check_glasses("current-frame.png")
    print("Does {} has glasses? {}".format(person_id, str(img_has_glasses)))

    if has_glasses[persons[person_id - 1]] and not img_has_glasses:
        if not no_glasses_alert:
            # alert the person to put on glasses
            no_glasses_alert = True
    else:
        if no_glasses_alert:
            # turn off the alert
            pass
        no_glasses_alert = False

    if person_id == "unknown":
        if not unknown_person_alert:
            # alert that person is unknown
            pass
    else:
        if unknown_person_alert:
            # turn off the alert
            pass
        unknown_person_alert = False

    curr_distance = get_distance()
    if curr_distance < 30:
        if not short_distance_alert:
            # alert that person is unknown
            pass
    else:
        if short_distance_alert:
            # turn off the alert
            pass
        short_distance_alert = False

    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
