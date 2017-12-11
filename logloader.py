import csv
import cv2
import numpy as np
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


# *** READ LOG *** #
logdir = './data/logs'
filename = 'LOG171211_131521.csv'
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
imgdir = './data/images/flyover2'
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

# *** DRAW PATH *** #
h = 800
w = 800
canvas = np.zeros((h, w, 3), dtype=np.uint8)
scale = 0.9
for i, p in enumerate(pose):
    if i == 0:
        continue
    x1 = int(round(w / 2 + scale * p[0]))
    y1 = int(round(h / 2 - scale * p[1]))
    x2 = int(round(w / 2 + scale * pose[i-1][0]))
    y2 = int(round(h / 2 - scale * pose[i-1][1]))
    cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow("canvas", canvas)
    cv2.imshow("img", images[i])
    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
