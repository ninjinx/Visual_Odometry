import cv2
import numpy as np
import math
import glob


def find_features(img, method=0):
    points = None
    time = 0
    e1 = cv2.getTickCount()
    if method == 0:
        points = cv2.goodFeaturesToTrack(img, mask=None, **good_params)
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
    elif method == 1:
        fast = cv2.FastFeatureDetector_create(**fast_params)
        kp = fast.detect(img, None)
        points = np.zeros((len(kp), 2), dtype=np.float32)
        for i, k in enumerate(kp):
            points[i] = [k.pt[0], k.pt[1]]
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
    return points, time


def remove_outliers(points1, points2, threshold=0.95):
    deltas = np.zeros(np.shape(points1), dtype=np.float32)
    for i, p1 in enumerate(points1):
        p2 = points2[i]
        s = np.subtract(p2, p1)
        # normalize deltas
        m = math.sqrt(s[0] ** 2 + s[1] ** 2)
        deltas[i] = [s[0] / m, s[1] / m]

    mdx = np.median(deltas[:, 0], overwrite_input=False)
    mdy = np.median(deltas[:, 1], overwrite_input=False)
    inliers = np.ones(len(deltas), dtype=np.uint8)
    for i, d in enumerate(deltas):
        ddot = d[0] * mdx + d[1] * mdy
        if ddot < threshold:
            inliers[i] = 0

    points1 = points1[inliers == 1]
    points2 = points2[inliers == 1]
    return points1, points2



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

fast_params = dict(threshold=21,
                   nonmaxSuppression=True,
                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)

imgdir = './data/images/flyover3'
img_filename = '*.bmp'  # image filename pattern. Use * followed by file extension

# *** READ FILENAMES *** #
images = []
filenames = []
timestamps = []
filepath = ''.join((imgdir, '/', img_filename))
for imgfile in glob.glob(filepath):
    filenames.append(imgfile)

filenames.sort()

for f in filenames:
    images.append(cv2.imread(f, 0))

h, w = np.shape(images[0])

m = 1


prev_img = None
n_points = 0
total_time = 0
inliers = 0
n_img = len(images)
for i, img in enumerate(images):
    print('{0:.2f}%'.format(100*(i/n_img)))
    if i == 0:
        prev_img = np.copy(img)
        continue

    points1, t1 = find_features(prev_img, method=m)
    points1 = np.reshape(points1, (-1, 1, 2))

    total_time += t1
    n_points += len(points1)

    points2, status, error = cv2.calcOpticalFlowPyrLK(prev_img, img, points1, None, **lk_params)
    points1 = points1[status == 1]
    points2 = points2[status == 1]

    points1, points2 = remove_outliers(points1, points2)
    inliers += len(points2)

    prev_img = np.copy(img)

n_points /= n_img
total_time /= n_img
inliers /= n_img

print('Method: {}'.format(m))
print('Images: {}'.format(n_img))
print('Avg. time taken: {}ms'.format(1000*total_time))
print('Avg. features found: {} '.format(n_points))
print('Avg. inliers: {} '.format(inliers))
print('Avg. inliers %: {}% '.format(100*(inliers/n_points)))

# composite = np.hstack((img1, img2))
# composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
# cv2.line(composite, (w, 0), (w, h), (0, 0, 0), 2)
# for i, p1 in enumerate(points1):
#     x1, y1 = p1
#     x2, y2 = points2[i]
#     x2 += w
#     cv2.line(composite, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
# cv2.imshow('correspondences', composite)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
