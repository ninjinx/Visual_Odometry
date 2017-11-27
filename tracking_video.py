import numpy as np
import cv2
import os
import time
from imutils.video import FPS
import math
import random

def get_speed(mat):
    vx = mat[0, 0]
    vy = mat[1, 0]
    vz = mat[2, 0]
    speed = vx**2+vy**2+vz**2
    speed = math.sqrt(speed)
    return speed


def get_rotation(mat, axis=0):
    if axis == 0:
        v0 = np.array([[1.], [0.], [0.]], dtype=np.float32)
    elif axis == 1:
        v0 = np.array([[0.], [1.], [0.]], dtype=np.float32)
    else:
        v0 = np.array([[0.], [0.], [1.]], dtype=np.float32)

    v1 = np.dot(mat, v0)
    dot = np.vdot(v0, v1)
    a = np.arccos(dot)
    return a


def draw_tracks(canvas, points1, points2):
    for i, (new, old) in enumerate(zip(points1, points2)):
        a, b = new.ravel()
        c, d = old.ravel()
        canvas = cv2.line(canvas, (a, b), (c, d), (0, 255, 0), 1)
    return canvas

min_features = 80

fps = FPS().start()

# image dimensions
h = 1080//2
w = 1920//2

canvas = np.zeros((h, w, 3), dtype=np.uint8)
path = np.zeros((h, w, 3), dtype=np.uint8)

# calibrated camera parameters
cam_mat = np.array([[949.38174439, 0., 479.90568209],
                    [0., 953.41397342, 261.82744181],
                    [0., 0., 1.]], dtype=np.float32)
dist_coeff = np.array((2.78460254e-01, -2.47492728e+00, -4.05126422e-04, 3.71587977e-03, 6.17755379e+00),
                      dtype=np.float32)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=5000,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

Rpos = np.eye(3, 3, dtype=np.float32)
Tpos = np.zeros((3, 1), dtype=np.float32)

cap = cv2.VideoCapture('VID_20171120_134746.mp4')
prev_frame = None
prev_gray = None

keyframe_old = None
keyframe_new = None
keypoint_dist = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if prev_frame is None:
        prev_frame = cv2.resize(frame, dsize=(w, h))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        keyframe_new = np.copy(prev_gray)
        continue

    frame = cv2.resize(frame, dsize=(w, h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    points1 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    points2, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points1, None, **lk_params)
    good_old = points1[status == 1]
    good_new = points2[status == 1]

    # calculate mean distance
    mean_dist = 0
    n = 0
    for i, p2 in enumerate(good_new):
        p1 = good_old[i]

        if p1 is None or p2 is None:
            continue

        if hasattr(p1, '__len__') is False:
            continue

        if hasattr(p2, '__len__') is False:
            continue

        if len(p1) <= 1 or len(p2) <= 1:
            continue

        dx = float(p1[0]) - float(p2[0])
        dy = float(p1[1]) - float(p2[1])
        mean_dist += math.sqrt(dx ** 2 + dy ** 2)
        n += 1

    if n > 0:
        mean_dist /= n
        keypoint_dist += mean_dist

    if keypoint_dist > 64:  # TODO change to variable
        keypoint_dist = 0
        keyframe_old = np.copy(keyframe_new)
        keyframe_new = np.copy(gray)

        points1 = cv2.goodFeaturesToTrack(keyframe_old, mask=None, **feature_params)
        points2, status, error = cv2.calcOpticalFlowPyrLK(keyframe_old, keyframe_new, points1, None, **lk_params)
        good_old = points1[status == 1]
        good_new = points2[status == 1]

        if len(good_new) > 7:
            good_new = np.reshape(np.array(good_new, dtype=np.float32), (-1, 1, 2))
            good_old = np.reshape(np.array(good_old, dtype=np.float32), (-1, 1, 2))
            E, mask = cv2.findEssentialMat(good_new, good_old,
                                           cameraMatrix=cam_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)
            points, R, T, newmask = cv2.recoverPose(E, good_new, good_old, cameraMatrix=cam_mat, mask=mask)

            old_pos = np.array(Tpos)
            Tpos = Tpos + np.dot(Rpos, T)
            Rpos = np.dot(R, Rpos)

            x1 = int(old_pos[0, 0] * 5 + w / 2)
            y1 = int(old_pos[1, 0] * 5 + h / 2)
            x2 = int(Tpos[0, 0] * 5 + w / 2)
            y2 = int(Tpos[1, 0] * 5 + h / 2)
            cv2.line(path, (x1, y1), (x2, y2), (0, 255, 0), 1)

    prev_gray = gray.copy()

    frame = cv2.add(frame, canvas)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Frame", frame)
    cv2.imshow("path", path)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        print("Finding new features..")
        points1 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

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
cap.release()
