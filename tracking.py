import numpy as np
import cv2
import os
import time
from imutils.video import VideoStream
from imutils.video import FPS
import math


class Feature:
    def __init__(self, x, y, patch):
        self.x = x
        self.y = y
        self.patch = patch

    def compare(self, patch):
        diff = cv2.absdiff(self.patch, patch)
        return np.mean(diff)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
fps = FPS().start()

frame = np.zeros((540, 960), dtype=np.uint8)
dsize = 16
dmax = 64

features = []

fast = cv2.FastFeatureDetector_create()
fast.setThreshold(30)

first_frame = True

canvas = np.zeros((540, 960), dtype=np.uint8)
prev_frame = np.zeros((540, 960, 3), dtype=np.uint8)
frame = np.array(prev_frame)

# loop over the frames from the video stream
while True:
    prev_frame = np.array(frame)
    frame = vs.read()
    diff = cv2.absdiff(frame, prev_frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, minDistance=16, useHarrisDetector=True)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    if corners is not None:
        new_features = []
        # kp = fast.detect(gray, None)
        matches = 0
        for c in corners:
            x = int(c[0][0])
            y = int(c[0][1])
            if x < dsize or x > w-dsize:
                continue

            if y < dsize or y > h-dsize:
                continue

            patch = gray[y - dsize//2:y + dsize//2, x - dsize//2:x + dsize//2]
            new_features.append(Feature(x, y, patch))

        if first_frame:
            first_frame = False
        else:
            for f1 in features:
                e_min = None
                match = None
                for f2 in new_features:
                    if math.fabs(f1.x-f2.x) > dmax:
                        continue

                    if math.fabs(f1.y-f2.y) > dmax:
                        continue

                    e = f1.compare(f2.patch)
                    if e_min is None:
                        e_min = e
                        match = f2
                        continue

                    if e < e_min:
                        e_min = e
                        match = f2

                if match is not None:
                    if e_min < 8.0:
                        matches += 1
                        dx = match.x-f1.x
                        dy = match.y-f1.y
                        d = math.sqrt(dx**2+dy**2)
                        if d < 64:
                            cv2.line(canvas, (match.x, match.y), (f1.x, f1.y), 255, 1)


                        black = np.ones((540, 960), dtype=np.uint8)*1
                        canvas = cv2.subtract(canvas, black)
                        #cv2.circle(frame, (match.x, match.y), dsize // 2, (255, 0, 0), 1)
        features = new_features
        print(matches)



    # show the output frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Diff", diff)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
