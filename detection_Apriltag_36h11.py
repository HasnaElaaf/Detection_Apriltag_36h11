from pvapriltags import Detector
import cv2
import numpy
from picamera2 import Picamera2
import os

# Configuration de l'acquisition et paramètres de la calibration
picam2 = Picamera2()
WIDTH, HEIGH = 3280, 2464
picam2.configure(picam2.create_preview_configuration({'size': (WIDTH, HEIGH)}))
picam2.start()

fx = 971.94324566
fy = 970.48731642
cx = 378.54097517
cy = 236.68125871
mtx = numpy.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist = numpy.array([[−0.583721964 8.69392525 −0.0311477149 0.0122369825 −36.3951426]])

# Acquisition de l’image, correction et détection des tags présents sur l’image
img = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_BGR2GRAY)  # prise d’une photo
at_detector = Detector()
img_undistorded = cv2.undistort(img, mtx, dist, None, newCameraMatrix=mtx)
tags = at_detector.detect(img_undistorded,
    estimate_tag_pose=True,
    camera_params=[fx, fy, cx, cy],
    tag_size=0.05)

print(tags)

# Affichage de l’image et pour chaque tags, des lignes reliant les coins et du numéro
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for tag in tags:
    for idx in range(len(tag.corners)):
        cv2.line(color_img,
            tuple(tag.corners[idx-1, :].astype(int)),
            tuple(tag.corners[idx, :].astype(int)),
            (0, 255, 0),
            5)

    cv2.putText(color_img,
        str(tag.tag_id),
        org=(tag.corners[0, 0].astype(int)+10, tag.corners[0, 1].astype(int)+10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(0, 0, 255),
        thickness=5)

cv2.imshow('Detected tags', color_img)
k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
