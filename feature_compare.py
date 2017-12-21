import cv2
import numpy as np
import math


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
good_params = dict(maxCorners=8000,
                      qualityLevel=0.03,
                      minDistance=5,
                      blockSize=3,
                      useHarrisDetector=False,
                      k=0.04)

fast_params = dict(threshold=15,
                   nonmaxSuppression=True,
                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)

img1 = cv2.imread('./data/images/flyover3/C171212_13580358.bmp', 0)
img2 = cv2.imread('./data/images/flyover3/C171212_13584265.bmp', 0)

m = 0
points1, t1 = find_features(img1, method=m)
points1 = np.reshape(points1, (-1, 1, 2))
npoints1 = len(points1)

print('Time taken for finding {} features using method {}: {}ms'.format(len(points1), m, t1 * 1000))

points2, status, error = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, **lk_params)
points1 = points1[status == 1]
points2 = points2[status == 1]

points1, points2 = remove_outliers(points1, points2)

npoints2 = len(points2)
print('{}, {}'.format(npoints1, npoints2))
print('{0:.2f}%'.format(100*(npoints2/npoints1)))

cv2.imshow('img', img1)
cv2.waitKey(1000)
cv2.imshow('img', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

