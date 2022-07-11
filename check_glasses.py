import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import statistics


def check_glasses(path="normal-img.png"):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    glasses_img = dlib.load_rgb_image(path)

    rect = detector(glasses_img)[0]
    sp = predictor(glasses_img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    nose_bridge_x = []
    nose_bridge_y = []

    for i in [28, 29, 30, 31, 33, 34, 35]:
        nose_bridge_x.append(landmarks[i][0])
        nose_bridge_y.append(landmarks[i][1])

    ### x_min and x_max
    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)

    ### ymin (from top eyebrow coordinate),  ymax
    y_min = landmarks[20][1]
    y_max = landmarks[31][1]

    nose_img = Image.open(path)
    nose_img = nose_img.crop((x_min, y_min, x_max, y_max))

    img_blur = cv2.GaussianBlur(np.array(nose_img), (3, 3), sigmaX=0, sigmaY=0)

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    plt.imshow(edges, cmap=plt.get_cmap('gray'))

    edges_center = edges.T[(int(len(edges.T) / 2))]

    if 255 in edges_center:
        # print("has glasses")
        return True
    # print("does not have glasses")
    return False


# check_glasses("glasses-img.png")
