import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
scale = 45.0 # [mm] real world size of checker board squares
chessboard_w = 6
chessboard_h = 8
objp = np.zeros((chessboard_w*chessboard_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_h, 0:chessboard_w].T.reshape(-1, 2)
objp = objp*scale

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./data/images/calibration/B*.bmp')
imshape = None

h = 580
w = 752
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
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()
ret, cam_mat, distcoeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imshape[::-1], None, None)
print(cam_mat)
print(distcoeff)