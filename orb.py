import numpy as np
import cv2
import os
import time


def load_sequence(f):
    s = []
    n = 0
    filename = f.format(n)
    while os.path.isfile(filename):
        img = cv2.imread(filename)
        img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
        s.append(img)
        n += 1
        filename = f.format(n)

    return s

sequence = load_sequence("frame{}.png")

fast = cv2.FastFeatureDetector_create()
fast.setThreshold(40)

descriptors1 = []
descriptors2 = []
kp1 = fast.detect(sequence[0], None)
kp2 = fast.detect(sequence[2], None)
dsize = 4

print(len(kp1))
print(len(kp2))

gray = cv2.cvtColor(sequence[0], cv2.COLOR_BGR2GRAY)
for k in kp1:
    center = (int(k.pt[0]), int(k.pt[1]))
    roi = gray[center[1]-dsize:center[1]+dsize, center[0]-dsize:center[0]+dsize]
    descriptors1.append(roi)

gray = cv2.cvtColor(sequence[2], cv2.COLOR_BGR2GRAY)
for k in kp2:
    center = (int(k.pt[0]), int(k.pt[1]))
    roi = gray[center[1]-dsize:center[1]+dsize, center[0]-dsize:center[0]+dsize]
    descriptors2.append(roi)

composite = np.vstack([sequence[0], sequence[1]])

vectors = []
h, w, channels = np.shape(sequence[0])

max_dist = 20

t0 = time.time()
for i1, k1 in enumerate(descriptors1):
    match = None
    e_min = None

    if np.prod(np.shape(k1)) == 0:
        continue

    for i2, k2 in enumerate(descriptors2):
        if np.shape(k1) != np.shape(k2):
            continue

        if np.prod(np.shape(k2)) == 0:
            continue

        if abs(kp1[i1].pt[0]-kp2[i2].pt[0]) > max_dist:
            continue

        if abs(kp1[i1].pt[1]-kp2[i2].pt[1]) > max_dist:
            continue

        dx = kp1[i1].pt[0] - kp2[i2].pt[0]

        diff = cv2.absdiff(k1, k2)
        e = np.mean(diff)

        if e_min is None:
            e_min = e
            match = i2
            continue

        if e < e_min:
            e_min = e
            match = i2

    if e_min is None:
        continue

    if e_min > 7:
        continue

    vectors.append((kp1[i1].pt[0], kp1[i1].pt[1], kp2[match].pt[0], kp2[match].pt[1]))

t1 = time.time()
print((t1-t0)*1000)
print(len(vectors))
for v in vectors:
    center1 = (int(v[0]), int(v[1]))
    center2 = (int(v[2]), int(v[3])+h)
    cv2.line(composite, center1, center2, (255, 0, 0), 1)

cv2.imshow("img", composite)
cv2.waitKey(0)

