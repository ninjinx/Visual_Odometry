import cv2
import numpy as np


cap = cv2.VideoCapture('VID_20171113_140000.mp4')

n = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("frame", frame)
    key = cv2.waitKey(40) & 0xFF

    if key == ord("s"):
        print("Saving image...")
        cv2.imwrite("v_img{0:03d}.png".format(n), frame)
        n += 1

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
