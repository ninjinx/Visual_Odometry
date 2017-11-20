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
        self.active = True

    def compare(self, patch):
        diff = cv2.absdiff(self.patch, patch)
        return np.mean(diff)


def find_features(im, dsize=8):
    feats = []
    corners = cv2.goodFeaturesToTrack(im, 200, 0.01, minDistance=dsize, useHarrisDetector=True)
    if corners is None:
        return []

    for c in corners:
        x = int(c[0][0])
        y = int(c[0][1])
        if x < dsize or x > w - dsize:
            continue

        if y < dsize or y > h - dsize:
            continue

        patch = im[y - dsize // 2:y + dsize // 2, x - dsize // 2:x + dsize // 2]
        feats.append(Feature(x, y, patch))

    return feats

dmax = 64
dsize = 21
min_features = 40

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
fps = FPS().start()

frame = vs.read()
# grab the frame dimensions and convert it to a blob
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
features = find_features(gray)
canvas = np.zeros((h, w, 3), dtype=np.uint8)
path = np.zeros((h, w, 3), dtype=np.uint8)

cam_mat = np.array([[733.89854196, 0., 520.27972203],
                    [0., 737.0390798, 254.92903388],
                    [0., 0., 1.]], dtype=np.float32)
dist_coeff = np.array((-0.13740599, 0.90600346, -0.0090014, 0.00825612, -1.70209085), dtype=np.float32)

Rpos = np.eye(3, 3, dtype=np.float32)
Tpos = np.zeros((3, 1), dtype=np.float32)

# loop over the frames from the video stream
while True:
    points1 = []
    points2 = []
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    matches = 0

    for f in features:
        if f.active is False:
            continue
        # do template matching
        roi = gray[f.y - dmax:f.y + dmax, f.x - dmax:f.x + dmax]
        roi_h, roi_w = np.shape(roi)
        if roi_h < dsize or roi_w < dsize:
            f.active = False
            continue
        res = cv2.matchTemplate(roi, f.patch, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.92:
            f.active = False
            continue

        xn, yn = max_loc
        xn += f.x-dmax
        yn += f.y-dmax
        xn += dsize//2
        yn += dsize//2

        cv2.line(canvas, (f.x, f.y), (xn, yn), (0, 255, 0), 1)
        black = np.ones((h, w, 3), dtype=np.uint8)
        canvas = cv2.subtract(canvas, black)

        points1.append((f.x, f.y))
        points2.append((xn, yn))
        f.x = xn
        f.y = yn
        f.patch = gray[yn - dsize // 2:yn + dsize // 2, xn - dsize // 2:xn + dsize // 2]
        matches += 1

    # if matches > 0:
    #     mx /= matches
    #     my /= matches
    #     vx -= mx
    #     vy -= my
    #     cv2.line(path, (int(vx+w/2), int(vy+h/2)), (int(vx+mx+w/2), int(vy+my+h/2)), (0, 255, 0), 1)

    if matches < min_features:
        print("Finding new features..")
        features = find_features(gray)

    if len(points1) > 7:
        points1 = np.reshape(np.array(points1, dtype=np.float32), (-1, 1, 2))
        points1 = cv2.undistortPoints(points1, cam_mat, dist_coeff)
        points2 = np.reshape(np.array(points2, dtype=np.float32), (-1, 1, 2))
        points2 = cv2.undistortPoints(points2, cam_mat, dist_coeff)
        E, mask = cv2.findEssentialMat(points1, points2, cam_mat, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        #retval, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
        #essential, mask = cv2.findEssentialMat(points1, points2, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        points, R, T, newmask = cv2.recoverPose(E, points1, points2, cam_mat)
        #C = np.vstack((np.hstack((r, t)), [0, 0, 0, 1]))
        x1 = float(Tpos[0, 0])*10
        y1 = float(Tpos[1, 0])*10

        Tpos = Tpos + np.dot(Rpos, T)
        Rpos = np.dot(R, Rpos)

        x2 = float(Tpos[0, 0])*10
        y2 = float(Tpos[1, 0])*10
        cv2.line(path, (int(x1 + w / 2), int(y1 + h / 2)), (int(x2 + w / 2), int(y2 + h / 2)), (0, 255, 0), 1)
        print(Tpos)

    # show the output frame
    frame = cv2.add(frame, canvas)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Frame", frame)
    cv2.imshow("path", path)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        print("Finding new features..")
        features = find_features(gray)

    if key == ord("r"):
        print("Resetting path...")
        Rpos = np.eye(3, 3, dtype=np.float32)
        Tpos = np.zeros((3, 1), dtype=np.float32)
        path = np.zeros((h, w, 3), dtype=np.uint8)

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
