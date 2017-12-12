import numpy as np
import cv2
import os
import time
from imutils.video import FPS
import math
import random
import csv
import glob
from datetime import datetime


def filename2unixtime(fn, houroffset=0):
    # extracts unix time from a given filename (and automatically removes path, prefixes and filetype from filename)
    # use houroffset to convert to different timezones

    # start by removing path and filetype from filename
    startind = fn.rfind('/')
    endind = fn.rfind('.')
    startind += 1

    if endind == -1:
        endind = len(fn)-1

    fn = fn[startind:endind]

    # remove prefixes
    while not fn[0].isdigit():
        fn = fn[1:]

    # remove sub-second resolution
    fn = fn[:-2]

    # extract date and time
    d = datetime.strptime(fn, '%y%m%d_%H%M%S')

    unixtime = d.timestamp()+3600*houroffset

    return unixtime


def get_speed(mat):
    newmat = np.array(np.reshape(mat, (3, 1)), dtype=np.float32)
    vx = newmat[0]
    vy = newmat[1]
    vz = newmat[2]
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


def get_mean_distance_2d(features1, features2):
    num_features = min((len(features1), len(features2)))
    features1 = np.reshape(features1, (num_features, 2))
    features2 = np.reshape(features2, (num_features, 2))

    features = zip(features1, features2)
    n = 0
    dist = 0
    for f1, f2 in features:
        dx = f1[0]-f2[0]
        dy = f1[1]-f2[1]
        d = math.sqrt(dx**2+dy**2)
        dist += d
        n += 1

    if n == 0:
        return 0

    dist /= n
    return dist


def update_motion(points1, points2, Rp, Tp, scale=1.0):
    p1 = np.reshape(np.array(points1, dtype=np.float32), (-1, 1, 2))
    p2 = np.reshape(np.array(points2, dtype=np.float32), (-1, 1, 2))
    E, mask = cv2.findEssentialMat(p1, p2,
                                   cameraMatrix=cam_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)
    points, R, T, newmask = cv2.recoverPose(E, p1, p2, cameraMatrix=cam_mat, mask=mask)

    Tp = Tp + np.dot(R, T)*scale
    Rp = np.dot(R, Rp)
    return Rp, Tp


def draw_path(cnv, points, R, rotate=True, drawVector=True, scale=0.1):
    dim = np.shape(cnv)
    cnv_h = dim[0]
    cnv_w = dim[1]

    #clear canvas
    cnv = np.zeros(dim, dtype=np.uint8)

    pos_final = points[-1]
    x_final = pos_final[0, 0]
    y_final = pos_final[1, 0]

    for i, p2 in enumerate(points):
        if i == 0:
            continue
        p1 = points[i-1]
        x1 = int((p1[0, 0] - x_final) * scale + cnv_w / 2)
        y1 = int((p1[1, 0] - y_final) * scale + cnv_h / 2)
        x2 = int((p2[0, 0] - x_final) * scale + cnv_w / 2)
        y2 = int((p2[1, 0] - y_final) * scale + cnv_h / 2)
        cv2.line(cnv, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if drawVector is True:
        vec = np.transpose(np.array([0., -1., 0.], dtype=np.float32))
        vec = np.dot(R, vec)
        cv2.line(cnv, (cnv_w//2, cnv_h//2),
                 (int(16*vec[0]+cnv_w/2), int(16*vec[1]+cnv_h/2)),
                 (255, 255, 255), 1)

    return cnv


def find_features(img, params, divisions=2):
    points = []

    dim = np.shape(img)

    if len(dim) > 2:  # if image has more than one channel then convert it to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h = dim[0]
    w = dim[1]
    dh = h//divisions
    dw = w//divisions

    for x in range(divisions):
        for y in range(divisions):
            partimg = gray[y*dh:(y+1)*dh, x*dw:(x+1)*dw]
            partpoints = cv2.goodFeaturesToTrack(partimg, mask=None, **feature_params)
            partpoints = np.reshape(partpoints, (-1, 2))
            for i in range(len(partpoints)):
                points.append([partpoints[i][0] + x * dw, partpoints[i][1] + y * dh])

    points = np.array(points, dtype=np.float32)
    points = np.reshape(points, (-1, 1, 2))
    return points


fps = FPS().start()

# image dimensions
h = 580
w = 752

canvas = np.zeros((h, w, 3), dtype=np.uint8)

# calibrated camera parameters
cam_mat = np.array([[1417.84363,   0.0,     353.360213],
                       [0.0,    1469.27135, 270.122002],
                       [0.0,       0.0,       1.0]], dtype=np.float32)
dist_coeff = np.array((-0.422318028, 0.495478798, 0.00205497237, 0.0000627845968, -2.67925535), dtype=np.float32)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=2000,
                      qualityLevel=0.01,
                      minDistance=5,
                      blockSize=3,
                      useHarrisDetector=False,
                      k=0.04)

Rpos = np.eye(3, 3, dtype=np.float32)
Tpos = np.zeros((3, 1), dtype=np.float32)

prev_frame = None
prev_gray = None

old_points = []
new_points = []

path = []

# *** READ LOG *** #
logdir = './data/logs'
filename = 'LOG171211_113343.csv'
file = open('/'.join((logdir, filename)), 'r')
logreader = csv.reader(file, delimiter=' ')

header = None
data = []

for i, row in enumerate(logreader):
    if i == 0:
        header = row
    else:
        numrow = []
        for dat in row:
            numrow.append(float(dat))
        data.append(numrow)

# *** READ FILENAMES *** #
imgdir = './data/images/flyover1'
filename = '*.bmp'
images = []
filenames = []
timestamps = []
filepath = ''.join((imgdir, '/', filename))
for imgfile in glob.glob(filepath):
    filenames.append(imgfile)

filenames.sort()

for f in filenames:
    timestamps.append(filename2unixtime(f))

# *** MATCH IMAGES TO LOGGED DATA *** #
pose = []
origin = [0, 0, 0]

logline = 0
for i, ts in enumerate(timestamps):
    while data[logline][0] < ts:
        logline += 1
        if logline >= len(data):
            break

    if logline >= len(data):
        break

    dat = data[logline]
    if i == 0:
        origin = [dat[4], dat[1] + dat[5], dat[6]]

    x = dat[4]
    y = dat[1] + dat[5]
    z = dat[6]

    x -= origin[0]
    y -= origin[1]
    z -= origin[2]
    pose.append([x, y, z])
    images.append(cv2.imread(filenames[i]))

for frame_num, frame in enumerate(images):
    if prev_frame is None:
        prev_frame = cv2.resize(frame, dsize=(w, h))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        continue

    frame = cv2.resize(frame, dsize=(w, h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # select new keypoints
    old_points = find_features(prev_gray, feature_params, divisions=2)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_points, None, **lk_params)
    old_points = old_points[status == 1]
    new_points = new_points[status == 1]

    old_pos = np.copy(Tpos)  # copy old position for drawing of path
    delta = np.subtract(pose[frame_num], pose[frame_num - 1])
    speed = get_speed(delta)
    print(speed)
    Rpos, Tpos = update_motion(new_points, old_points, Rpos, Tpos, scale=speed)

    # draw path
    path.append(Tpos)
    canvas = draw_path(canvas, path, Rpos, scale=0.6)

    # draw tracks
    frame = draw_tracks(frame, old_points, new_points)

    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv2.imshow("Frame", frame)
    cv2.imshow("Path", canvas)

    prev_gray = gray.copy()
    key = cv2.waitKey(100) & 0xFF

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
