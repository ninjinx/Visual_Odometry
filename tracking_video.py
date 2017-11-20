import numpy as np
import cv2
import os
import time
from imutils.video import FPS
import math
import random


def get_scale(p1, p2, iterations=1000):
    if p1 is None or p2 is None:
        return None

    if len(p1) <= 1 or len(p2) <= 1:
        return None

    n = min((len(p1), len(p2)))-1
    ratio = 0
    for i in range(iterations):
        ind1 = random.randint(0, n)
        ind2 = random.randint(0, n)
        while ind2 == ind1:
            ind2 = random.randint(0, n)

        if len(np.shape(p1)) > 2:
            d1x = p1[ind1, 0, 0]-p1[ind2, 0, 0]
            d1y = p1[ind1, 0, 1]-p1[ind2, 0, 1]
        else:
            d1x = p1[ind1, 0] - p1[ind2, 0]
            d1y = p1[ind1, 1] - p1[ind2, 1]
        d1 = math.sqrt(d1x**2+d1y**2)

        if len(np.shape(p1)) > 2:
            d2x = p2[ind1, 0, 0] - p2[ind2, 0, 0]
            d2y = p2[ind1, 0, 1] - p2[ind2, 0, 1]
        else:
            d2x = p2[ind1, 0] - p2[ind2, 0]
            d2y = p2[ind1, 1] - p2[ind2, 1]
        d2 = math.sqrt(d2x ** 2 + d2y ** 2)

        if d1 == 0:
            ratio += 1
            continue

        ratio += d2/d1

    ratio /= iterations
    return ratio


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

dmax = 32
dsize = 16
min_features = 300

print("[INFO] starting video stream...")
fps = FPS().start()

h = 1080//2
w = 1920//2
features = []
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
Vpos = np.zeros((3, 1), dtype=np.float32)
Rvpos = np.eye(3, 3, dtype=np.float32)

cap = cv2.VideoCapture('VID_20171120_134746.mp4')
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, dsize=(w, h))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

points1 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

frame_n = 0

scale = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, dsize=(w, h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optical flow tracking
    while points1 is None:
        points1 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    points2, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points1, None, **lk_params)
    status[error > 5.] = 0   # error rejection
    good_new = points2[status == 1]
    good_old = points1[status == 1]

    # draw tracks
    black = np.ones((h, w, 3), dtype=np.uint8)*30
    canvas = cv2.subtract(canvas, black)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        canvas = cv2.line(canvas, (a, b), (c, d), (0, 255, 0), 1)

    frame = cv2.add(frame, canvas)

    matches = len(good_new)

    if matches < min_features:
        #print("Finding new features.. {}".format(frame_n))
        points1 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    if matches > 7:
        good_new = np.reshape(np.array(good_new, dtype=np.float32), (-1, 1, 2))
        #good_new = cv2.undistortPoints(good_new, cam_mat, dist_coeff)
        good_old = np.reshape(np.array(good_old, dtype=np.float32), (-1, 1, 2))
        #good_old = cv2.undistortPoints(good_old, cam_mat, dist_coeff)
        #E, mask = cv2.findEssentialMat(good_new, good_old, cam_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        E, mask = cv2.findEssentialMat(good_new, good_old,
                                       cameraMatrix=cam_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)
        points, R, T, newmask = cv2.recoverPose(E, good_new, good_old, cameraMatrix=cam_mat, mask=mask)

        if np.rad2deg(get_rotation(R)-get_rotation(Rvpos)) < 3.:
            Vpos = T
            Rvpos = R

    new_scale = get_scale(good_old, good_new)
    if scale is None:
        if new_scale is not None:
            scale = new_scale
        else:
            scale = 1.
    elif new_scale is not None:
            scale *= new_scale
    print(scale)
    old_pos = np.array(Tpos)
    Tpos = Tpos + scale*np.dot(Rpos, Vpos)
    Rpos = np.dot(Rvpos, Rpos)

    x1 = int(old_pos[0, 0]*5+w/2)
    y1 = int(old_pos[1, 0]*5+h/2)
    x2 = int(Tpos[0, 0]*5+w/2)
    y2 = int(Tpos[1, 0]*5+h/2)
    cv2.line(path, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # draw view vector
    # v = np.array([[0.], [0.], [1.]], dtype=np.float32)
    # v = np.dot(Rpos, v)
    # path = np.zeros((h, w, 3), dtype=np.uint8)
    # cv2.line(path, (int(w/3), int(h / 2)),
    #                (int(w/3 + v[1][0]*128), int(h / 2 + v[2][0]*128)), (0, 255, 0), 1)
    #
    # cv2.line(path, (int(2*w / 3), int(h / 2)),
    #          (int(2*w / 3 + v[0][0] * 128), int(h / 2 + v[2][0] * 128)), (0, 255, 0), 1)

    #print("{}, {}, {}".format(Tpos[0, 0], Tpos[1, 0], Tpos[2, 0]))
    #print(np.dot(Rpos, np.transpose(np.array([1., 0., 0.], dtype=np.float32))))

    prev_gray = gray.copy()

    # show the output frame
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
    frame_n += 1

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
