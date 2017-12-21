##################################
######## surface_scanner.py ######
##################################
######### Copyright 2017 #########
##################################
####### Thomas T. Sørensen #######
######## Hjalte B. Møller ########
####### Mads Z. Mackeprang #######
##################################
##################################
#  Generates a pointcloud from   #
#  an image sequence and saves   #
#  it to a .csv file.            #
#  A log file with the camera    #
#  position is required.         #
##################################

import numpy as np
import cv2
from imutils.video import FPS
import math
import csv
import glob
from datetime import datetime

### CONFIG ###
save_as = 'pointcloud_fixed.csv'  # filename of point-cloud
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
feature_params = dict(maxCorners=2000,
                      qualityLevel=0.01,
                      minDistance=5,
                      blockSize=3,
                      useHarrisDetector=False,
                      k=0.04)

# calibrated camera parameters
# obtained from calibrate_camera.py
cam_mat = np.array([[1417.84363,   0.0,     353.360213],
                       [0.0,    1469.27135, 270.122002],
                       [0.0,       0.0,       1.0]], dtype=np.float32)
dist_coeff = np.array((-0.422318028, 0.495478798, 0.00205497237, 0.0000627845968, -2.67925535), dtype=np.float32)
##############


def undistort(points, cm, dc):
    distorted_points = np.array(points, dtype=np.float32)
    distorted_points = np.reshape(distorted_points, (-1, 1, 2))
    undistorted_points = cv2.undistortPoints(distorted_points, cameraMatrix=cm, distCoeffs=dc)
    return undistorted_points


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


def draw_tracks(canvas, points1, points2, mask=None):
    if mask is None:
        points1_in = points1
        points2_in = points2
    else:
        points1_in = points1[inliers == 1]
        points2_in = points2[inliers == 1]
        points1_out = points1[inliers == 0]
        points2_out = points2[inliers == 0]

        for i, (new, old) in enumerate(zip(points1_out, points2_out)):
            a, b = new.ravel()
            c, d = old.ravel()
            canvas = cv2.line(canvas, (a, b), (c, d), (0, 0, 255), 2)

    for i, (new, old) in enumerate(zip(points1_in, points2_in)):
        a, b = new.ravel()
        c, d = old.ravel()
        canvas = cv2.line(canvas, (a, b), (c, d), (0, 255, 0), 1)
    return canvas


def find_features(img, params, divisions=2):
    # divides image into sub images, finds features in each image and returns the features
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
            partpoints = cv2.goodFeaturesToTrack(partimg, mask=None, **params)
            partpoints = np.reshape(partpoints, (-1, 2))
            for i in range(len(partpoints)):
                points.append([partpoints[i][0] + x * dw, partpoints[i][1] + y * dh])

    points = np.array(points, dtype=np.float32)
    points = np.reshape(points, (-1, 1, 2))
    return points

fps = FPS().start()

prev_frame = None
prev_gray = None

old_points = []
new_points = []

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
    images.append(cv2.imread(filenames[i]))

csv_file = open(save_as, 'w')

for frame_num, frame in enumerate(images):
    if prev_frame is None:
        prev_frame = cv2.resize(frame, dsize=(w, h))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        continue

    print('Frame number {}:'.format(frame_num))

    frame = cv2.resize(frame, dsize=(w, h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # select new keypoints
    old_points = find_features(prev_gray, feature_params, divisions=4)

    # track points
    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_points, None, **lk_params)
    old_points = old_points[status == 1]
    new_points = new_points[status == 1]

    # get speed from log data
    delta = np.subtract(pose[frame_num], pose[frame_num - 1])
    speed = get_speed(delta)

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

    # reconstruct 3D point cloud
    if speed > 4.0:
        # set up camera projection matrices
        R = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 1]], dtype=np.float32)
        T1 = np.reshape(pose[frame_num - 1], (3, 1))
        T2 = np.reshape(pose[frame_num], (3, 1))
        T1 = np.dot(R, T1)
        T2 = np.dot(R, T2)
        R = np.eye(3, 3, dtype=np.float32)

        RT1 = np.hstack((R, T1))
        RT2 = np.hstack((R, T2))

        dp1 = np.reshape(old_points, (-1, 2))
        dp2 = np.reshape(new_points, (-1, 2))
        udp1 = undistort(dp1, cam_mat, dist_coeff)
        udp2 = undistort(dp2, cam_mat, dist_coeff)

        points4D = cv2.triangulatePoints(RT1, RT2, udp1, udp2)
        points3D = np.array([points4D[0] / points4D[3], points4D[1] / points4D[3], points4D[2] / points4D[3]],
                            dtype=np.float32)
        points3D = np.transpose(points3D)
        for p in points3D:
            x, y, z = p
            if z < -800 or z > -660:
                continue
            csv_file.write('{0:f}, {1:f}, {2:f}\n'.format(x, y, z))

    frame = draw_tracks(frame, old_points, new_points)

    cv2.imshow("Frame", frame)

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
cv2.destroyAllWindows()
csv_file.close()
