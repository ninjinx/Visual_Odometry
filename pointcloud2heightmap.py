import numpy as np
import cv2
from imutils.video import FPS
import math
import csv
import glob
from datetime import datetime
from skimage.filters.rank import median

filename = './pointcloud_fixed.csv'
file = open(filename, 'r')
logreader = csv.reader(file, delimiter=',')

data = []

for i, row in enumerate(logreader):
    numrow = []
    for dat in row:
        numrow.append(float(dat))
    data.append(numrow)
file.close()

print('Loading csv complete')

xdat = []
ydat = []
zdat = []

for d in data:
    x, y, z = d
    xdat.append(x)
    ydat.append(y)
    zdat.append(z)

minx = float(np.min(xdat))
maxx = float(np.max(xdat))
miny = float(np.min(ydat))
maxy = float(np.max(ydat))
minz = float(np.min(zdat))
maxz = float(np.max(zdat))

w = int(maxx-minx)
h = int(maxy-miny)
zrange = maxz-minz

print('Extracting data complete')
heightmap = np.zeros((h, w), dtype=np.float32)
points_per_pixel = np.zeros((h, w), dtype=np.uint16)
for x1, y1, z1 in data:
    if z1-minz > 200:
        continue
    xi = round(w * ((x1 - minx) / (maxx - minx)))
    yi = round(h * ((y1 - miny) / (maxy - miny)))
    xi = max((min((xi, w-1)), 0))
    yi = max((min((yi, h - 1)), 0))
    val = (z1-minz)/zrange
    heightmap[yi, xi] += val
    points_per_pixel[yi, xi] += 1

print('Filling in heightmap complete')
for yi in range(h):
    for xi in range(w):
        p = points_per_pixel[yi, xi]
        if p > 0:
            heightmap[yi, xi] /= p

print('Median filtering..')
canvas = np.array(heightmap*255, dtype=np.uint8)
mask = np.zeros(np.shape(canvas), dtype=np.uint8)
mask[points_per_pixel > 0] = 255
canvas = median(canvas, mask=points_per_pixel)

print(np.mean(points_per_pixel))

# print('Filling in holes..')
# canvas = cv2.inpaint(canvas, cv2.bitwise_not(mask), 3, cv2.INPAINT_TELEA)
cv2.imshow('canvas', canvas)
cv2.imwrite('heightmap_fixed.png', canvas)
print(maxz-minz)
cv2.waitKey(0)
