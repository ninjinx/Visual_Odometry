##################################
###### calibrate_camera.py ######
##################################
##################################
####### Thomas T. Sørensen #######
######## Hjalte B. Møller ########
####### Mads Z. Mackeprang #######
##################################
##################################
#  Does camera calibration using #
#  chessboard. Input a sequence  #
#  of images of a calibration    #
#  chessboard at different       #
#  angles and positions in the   #
#  image and this script will    #
#  output the cameras intrinsic  #
#  parameters and distortion     #
#  coefficients.                 #
##################################

#  Adapted from https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html #

import cv2
import numpy as np
import glob

### CONFIG ###
# image dimensions
h = 580
w = 752
scale = 45.0  # [mm] real world size of checker board squares
# number of chessboard squares:
chessboard_w = 6
chessboard_h = 8
##############

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboard_w*chessboard_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_h, 0:chessboard_w].T.reshape(-1, 2)
objp = objp*scale

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('./data/images/calibration/B*.bmp')
imshape = None

for fname in images:
    img = cv2.imread(fname)
    if (h, w, 3) != np.shape(img):
        img = cv2.resize(img, dsize=(w, h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imshape = gray.shape

    ret, corners = cv2.findChessboardCorners(gray, (chessboard_h, chessboard_w), None)

    if ret is True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (chessboard_h, chessboard_w), corners, ret)
        cv2.imshow('img', img)
        fname_new = fname[0:-4]+'_ann.bmp'
        cv2.imwrite(fname_new, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()
ret, cam_mat, distcoeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imshape[::-1], None, None)
print(cam_mat)
print(distcoeff)
