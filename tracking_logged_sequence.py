##################################
## tracking_logged_sequence.py ###
##################################
######### Copyright 2017 #########
##################################
####### Thomas T. Sørensen #######
######## Hjalte B. Møller ########
####### Mads Z. Mackeprang #######
##################################
##################################
#  Does visual odometry from an  #
#  image sequence. A log file    #
#  containing ground truth data  #
#  is required to get correct    #
#  scale.                        #
##################################

import numpy as np
import cv2
from imutils.video import FPS
import math
import glob
from datetime import datetime
import csv

### CONFIG ###
save_as = 'path.csv'  # filename of path
logdir = './data/logs'
log_filename = 'LOG171212_135705.csv'
imgdir = './data/images/flyover3'
img_filename = '*.bmp'  # image filename pattern. Use * followed by file extension
# image dimensions
h = 580
w = 752

min_features = 600

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# params for ShiTomasi corner detection
good_params = dict(maxCorners=2000,
                      qualityLevel=0.01,
                      minDistance=5,
                      blockSize=3,
                      useHarrisDetector=False,
                      k=0.04)
# params for FAST detection
fast_params = dict(threshold=15,
                   nonmaxSuppression=True,
                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)

# calibrated camera parameters
# obtained from calibrate_camera.py
cam_mat = np.array([[1417.84363,   0.0,     353.360213],
                       [0.0,    1469.27135, 270.122002],
                       [0.0,       0.0,       1.0]], dtype=np.float32)
dist_coeff = np.array((-0.422318028, 0.495478798, 0.00205497237, 0.0000627845968, -2.67925535), dtype=np.float32)
##############


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


def draw_tracks(canvas, points1, points2):
    for i, (new, old) in enumerate(zip(points1, points2)):
        a, b = new.ravel()
        c, d = old.ravel()
        canvas = cv2.line(canvas, (a, b), (c, d), (0, 255, 0), 1)
    return canvas


def update_motion(points1, points2, Rp, Tp, scale=1.0):
    p1 = np.reshape(np.array(points1, dtype=np.float32), (-1, 1, 2))
    p2 = np.reshape(np.array(points2, dtype=np.float32), (-1, 1, 2))
    E, mask = cv2.findEssentialMat(p1, p2,
                                   cameraMatrix=cam_mat, method=cv2.LMEDS, prob=0.999, threshold=1.0, mask=None)
    points, R, T, newmask = cv2.recoverPose(E, p1, p2, cameraMatrix=cam_mat, mask=mask)

    Tp = Tp + np.dot(Rp, T)*scale
    Rp = np.dot(R, Rp)
    return Rp, Tp


def draw_path(cnv, points, rotate=0,
              scale=0.1,
              clear=True,
              color=(255, 255, 255),
              flipX=False,
              flipY=False):
    points = np.reshape(points, (-1, 3))
    dim = np.shape(cnv)
    cnv_h = dim[0]
    cnv_w = dim[1]

    #clear canvas
    if clear:
        cnv = np.zeros(dim, dtype=np.uint8)

    for n in range(rotate):
        for i, p in enumerate(points):
            points[i] = [-p[1], p[0], p[2]]

    if flipX:
        for i, p in enumerate(points):
            points[i] = [-p[0], p[1], p[2]]

    if flipY:
        for i, p in enumerate(points):
            points[i] = [p[0], -p[1], p[2]]

    pos_orig = points[0]
    x_orig = pos_orig[0]
    y_orig = pos_orig[1]

    for i, p2 in enumerate(points):
        if i == 0:
            continue
        p1 = points[i-1]
        x1 = int((p1[0] - x_orig) * scale + cnv_w / 2)
        y1 = int((p1[1] - y_orig) * scale + cnv_h / 2)
        x2 = int((p2[0] - x_orig) * scale + cnv_w / 2)
        y2 = int((p2[1] - y_orig) * scale + cnv_h / 2)
        cv2.line(cnv, (x1, y1), (x2, y2), color, 1)

    return cnv


def find_features(img, params, divisions=2, method=0):
    # divides image into sub images, finds features in each image and returns the features
    points = []
    fast = cv2.FastFeatureDetector_create(**fast_params)

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
            if method == 0:
                partpoints = cv2.goodFeaturesToTrack(partimg, mask=None, **params)
            else:
                kp = fast.detect(img, None)
                partpoints = np.zeros((len(kp), 2), dtype=np.float32)
                for i, k in enumerate(kp):
                    partpoints[i] = [k.pt[0], k.pt[1]]
            partpoints = np.reshape(partpoints, (-1, 2))
            for i in range(len(partpoints)):
                points.append([partpoints[i][0] + x * dw, partpoints[i][1] + y * dh])

    points = np.array(points, dtype=np.float32)
    points = np.reshape(points, (-1, 1, 2))
    return points


fps = FPS().start()

Rpos = np.eye(3, 3, dtype=np.float32)
Tpos = np.zeros((3, 1), dtype=np.float32)

prev_frame = None
prev_gray = None

old_points = []
new_points = []

path = []

# *** READ LOG *** #
file = open('/'.join((logdir, log_filename)), 'r')
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
file.close()

# *** READ FILENAMES *** #
images = []
filenames = []
timestamps = []
filepath = ''.join((imgdir, '/', img_filename))
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
    img = cv2.imread(filenames[i])
    images.append(img)


# *** VISUAL ODOMETRY *** #
csv_file = open(save_as, 'w')
canvas = np.zeros((h, w, 3), dtype=np.uint8)
for frame_num, frame in enumerate(images):
    if prev_frame is None:
        prev_frame = cv2.resize(frame, dsize=(w, h))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        continue

    frame = cv2.resize(frame, dsize=(w, h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # select new keypoints
    old_points = find_features(prev_gray, good_params, divisions=4)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_points, None, **lk_params)
    old_points = old_points[status == 1]
    new_points = new_points[status == 1]

    # remove outliers
    deltas = np.zeros(np.shape(old_points), dtype=np.float32)
    for i, p1 in enumerate(old_points):
        p2 = new_points[i]
        s = np.subtract(p2, p1)
        # normalize deltas
        m = math.sqrt(s[0] ** 2 + s[1] ** 2)
        deltas[i] = [s[0] / m, s[1] / m]

    mdx = np.median(deltas[:, 0], overwrite_input=False)
    mdy = np.median(deltas[:, 1], overwrite_input=False)
    thres = 0.9
    inliers = np.ones(len(deltas), dtype=np.uint8)
    for i, d in enumerate(deltas):
        ddot = d[0] * mdx + d[1] * mdy
        if ddot < thres:
            inliers[i] = 0

    if np.sum(inliers) > min_features:
        deltas = deltas[inliers == 1]
        old_points = old_points[inliers == 1]
        new_points = new_points[inliers == 1]

    old_pos = np.copy(Tpos)  # copy old position for drawing of path
    delta = np.subtract(pose[frame_num], pose[frame_num - 1])
    speed = get_speed(delta)
    _, Tpos = update_motion(new_points, old_points, Rpos, Tpos, scale=speed)


    # draw path
    gtx, gty, gtz = pose[frame_num]
    tx, ty, tz = Tpos[:]
    csv_file.write('{0:f}, {1:f}, {2:f}, {3:f}, {4:f}, {5:f}\n'.format(float(tx), float(ty), float(tz), gtx, gty, gtz))
    path.append(Tpos)
    canvas = draw_path(canvas, path, scale=0.6, clear=True, color=(0, 255, 0))
    canvas = draw_path(canvas, pose[0:frame_num],
                       rotate=1,
                       scale=0.6,
                       clear=False,
                       color=(255, 0, 0),
                       flipX=True)

    # draw tracks
    frame = draw_tracks(frame, old_points, new_points)

    cv2.imshow("Frame", frame)
    cv2.imshow("Path", canvas)

    prev_gray = gray.copy()
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.waitKey(0)
cv2.destroyAllWindows()
csv_file.close()
