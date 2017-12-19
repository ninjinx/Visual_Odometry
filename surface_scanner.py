import numpy as np
import cv2
from imutils.video import FPS
import math
import csv
import glob
from datetime import datetime


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


def draw_path(cnv, points, R, rotate=0,
              drawVector=True,
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

    pos_final = points[-1]
    x_final = pos_final[0]
    y_final = pos_final[1]

    for i, p2 in enumerate(points):
        if i == 0:
            continue
        p1 = points[i-1]
        x1 = int((p1[0] - x_final) * scale + cnv_w / 2)
        y1 = int((p1[1] - y_final) * scale + cnv_h / 2)
        x2 = int((p2[0] - x_final) * scale + cnv_w / 2)
        y2 = int((p2[1] - y_final) * scale + cnv_h / 2)
        cv2.line(cnv, (x1, y1), (x2, y2), color, 1)

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


def draw_point_cloud(cnv, points, clear=True, scale=1.0):
    dim = np.shape(cnv)
    cnv_h = dim[0]
    cnv_w = dim[1]

    # clear canvas
    if clear:
        cnv = np.zeros(dim, dtype=np.uint8)

    nonzeropoints = []
    for p in points:
        if p[0] == 0 and p[1] == 0 and p[2] == 0:
            continue

        nonzeropoints.append(p)

    if len(nonzeropoints) == 0:
        return cnv

    #print(np.max(nonzeropoints[:][2]))
    for p in nonzeropoints:
        x, y, z = p
        print('{}, {}, {}'.format(x, y, z))

    return cnv


csv_file = open('pointcloud_new.csv', 'w')

fps = FPS().start()

# image dimensions
h = 580
w = 752

min_features = 600

canvas = np.zeros((h, w, 3), dtype=np.uint8)

# calibrated camera parameters
cam_mat = np.array([[1417.84363,   0.0,     353.360213],
                       [0.0,    1469.27135, 270.122002],
                       [0.0,       0.0,       1.0]], dtype=np.float32)
dist_coeff = np.array((-0.422318028, 0.495478798, 0.00205497237, 0.0000627845968, -2.67925535), dtype=np.float32)


# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                 minEigThreshold=1e-2)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=2000,
                      qualityLevel=0.01,
                      minDistance=5,
                      blockSize=3,
                      useHarrisDetector=False,
                      k=0.04)

prev_frame = None
prev_gray = None
heightmap = np.zeros((800, 800), dtype=np.uint8)

old_points = []
new_points = []

# *** READ LOG *** #
logdir = './data/logs'
filename = 'LOG171212_135705.csv'
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
file.close()

# *** READ FILENAMES *** #
imgdir = './data/images/flyover3'
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
    old_points = find_features(prev_gray, feature_params, divisions=4)

    # track points
    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_points, None, **lk_params)
    old_points = old_points[status == 1]
    new_points = new_points[status == 1]

    # remove outliers
    delta = np.subtract(pose[frame_num], pose[frame_num - 1])
    speed = get_speed(delta)

    deltas = np.zeros(np.shape(old_points), dtype=np.float32)
    for i, p1 in enumerate(old_points):
        p2 = new_points[i]
        s = np.subtract(p2, p1)
        deltas[i] = s

    mdx = np.median(deltas[:, 0], overwrite_input=False)
    mdy = np.median(deltas[:, 1], overwrite_input=False)
    mx = np.mean(deltas[:, 0])
    my = np.mean(deltas[:, 1])
    mlength = math.sqrt((mdx - mx)**2 + (mdy - my)**2)
    thres = mlength * 1.8
    inliers = np.ones(len(deltas), dtype=np.uint8)
    for i, d in enumerate(deltas):
        ex = d[0]-mdx
        ey = d[1]-mdy
        e = math.sqrt(ex**2+ey**2)
        if e > thres:
            inliers[i] = 0

    if np.sum(inliers) > min_features:
        deltas = deltas[inliers == 1]
        old_points = old_points[inliers == 1]
        new_points = new_points[inliers == 1]

    # reconstruct 3D point cloud
    if speed > 4.0:
        # set up camera projection matrices
        R = np.array([[0, 1, 0],
             [-1, 0, 0],
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
            if z > 800 or z < 660:
                continue
            csv_file.write('{0:f}, {1:f}, {2:f}\n'.format(x, y, z))

        # for p in points3D:
        #     px, py, pz = np.add(p, pose[frame_num - 1])
        #     csv_file.write('{0:f}, {1:f}, {2:f}\n'.format(px, py, pz))

    # draw path
    canvas = draw_path(canvas, pose[0:frame_num], None,
                       rotate=1,
                       scale=0.6,
                       clear=True,
                       drawVector=False,
                       color=(255, 0, 0),
                       flipX=True)

    # draw tracks
    frame = draw_tracks(frame, old_points, new_points)

    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv2.line(frame, (0, h // 4), (w, h // 4), (255, 0, 0), 1)
    cv2.line(frame, (w // 4, 0), (w // 4, h), (255, 0, 0), 1)
    cv2.line(frame, (0, 3 * h // 4), (w, 3 * h // 4), (255, 0, 0), 1)
    cv2.line(frame, (3 * w // 4, 0), (3 * w // 4, h), (255, 0, 0), 1)
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
cv2.waitKey(0)
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
csv_file.close()