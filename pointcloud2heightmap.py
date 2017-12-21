##################################
##### pointcloud2heightmap.py ####
##################################
######### Copyright 2017 #########
##################################
####### Thomas T. Sørensen #######
######## Hjalte B. Møller ########
####### Mads Z. Mackeprang #######
##################################
##################################
#  Converts pointcloud data to a #
#  heightmap image. Useful for   #
#  creating 3D mesh from point-  #
#  cloud data. Pointclouds can   #
#  be obtained using the script  #
#  surface_scanner.py ############
##################################
import numpy as np
import cv2
import csv
from skimage.filters.rank import median

### CONFIG ###
filename = './pointcloud_fixed.csv'
inpaint = False  # whether or not to fill in missing data
export = False  # whether or not to save the heightmap to disk
save_as = 'heightmap.png'  # filename of output file. Only used if 'export' is set to True
px_per_mm = 2.0
##############


def load_csv(filename):
    file = open(filename, 'r')
    logreader = csv.reader(file, delimiter=',')

    data = []

    for i, row in enumerate(logreader):
        numrow = []
        for dat in row:
            numrow.append(float(dat))
        data.append(numrow)
    file.close()

    return data


def extract_coords(data):
    xdat = []
    ydat = []
    zdat = []

    for d in data:
        x, y, z = d
        xdat.append(x)
        ydat.append(y)
        zdat.append(z)

    return xdat, ydat, zdat


def generate_heightmap(data, filterImg=True, verbose=True, resolution=1.0):
    if verbose:
        print('Extracting data...')
    xdat, ydat, zdat = extract_coords(data)
    # extract canvas size
    minx = float(np.min(xdat))
    maxx = float(np.max(xdat))
    miny = float(np.min(ydat))
    maxy = float(np.max(ydat))
    minz = float(np.min(zdat))
    maxz = float(np.max(zdat))

    w = int((maxx - minx)*resolution)
    h = int((maxy - miny)*resolution)
    zrange = maxz - minz

    if verbose:
        print('Generating heightmap...')

    heightmap = np.zeros((h, w), dtype=np.float32)
    points_per_pixel = np.zeros((h, w), dtype=np.uint16)
    for x1, y1, z1 in data:
        if z1 - minz > 200:
            continue
        xi = round(w * ((x1 - minx) / (maxx - minx)))
        yi = round(h * ((y1 - miny) / (maxy - miny)))
        xi = max((min((xi, w - 1)), 0))
        yi = max((min((yi, h - 1)), 0))
        val = (z1 - minz) / zrange
        heightmap[yi, xi] += val
        points_per_pixel[yi, xi] += 1

    for yi in range(h):
        for xi in range(w):
            p = points_per_pixel[yi, xi]
            if p > 0:
                heightmap[yi, xi] /= p

    canvas = np.array(heightmap * 255, dtype=np.uint8)
    mask = np.zeros(np.shape(canvas), dtype=np.uint8)
    mask[points_per_pixel > 0] = 255

    if filterImg:
        if verbose:
            print('Median filtering heightmap...')
        canvas = median(canvas, mask=points_per_pixel)

    if verbose:
        print('Z-scale: {0:.2f}mm'.format(zrange))

    return canvas, mask

print('Loading csv...')
data = load_csv(filename)
canvas, mask = generate_heightmap(data, resolution=px_per_mm)

if inpaint:
    print('Filling in holes..')
    canvas = cv2.inpaint(canvas, cv2.bitwise_not(mask), 3, cv2.INPAINT_TELEA)
cv2.imshow('canvas', canvas)
cv2.imwrite(save_as, canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
