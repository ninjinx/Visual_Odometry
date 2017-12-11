import cv2
import numpy as np
from imutils.video import VideoStream

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

n = 0

while True:
    frame = vs.read()

    cv2.imshow("img", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("Saving image...")
        cv2.imwrite("img{0:03d}.png".format(n), frame)
        n += 1

    if key == ord("q"):
        break
