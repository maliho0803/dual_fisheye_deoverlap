import cv2
import numpy as np
import glob

#rectify fisheye image
def get_distort_map(K, D, R, P, size=(1280, 960), alpha=0.0, fov_scale=1):
    f = (K[0][0], K[1][1])
    c = (K[0][2], K[1][2])

    if P.shape[1] == 4:
        PP = P[:, 0:3]
    else:
        PP = P
    #print("PP = " + str(PP))
    mm = np.mat(PP) * np.mat(R)
    _, iR = cv2.invert(mm, flags=cv2.DECOMP_SVD)

    tempMatrix = np.zeros((size[1], size[0], 2), np.float32)

    def _calc_iR(idx, size, iR):
        _x = np.tile(np.arange(size[1]) * iR[idx][1] + iR[idx][2], size[0]).reshape((-1, size[1])).T
        _accum_x = np.arange(size[0]) * iR[idx][0]
        return np.apply_along_axis(lambda x: x + _accum_x, 1, _x)

    #get camera coordinates point
    _x = _calc_iR(0, size, iR)
    _y = _calc_iR(1, size, iR)
    _w = _calc_iR(2, size, iR)
    x = _x / _w
    y = _y / _w
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r / fov_scale)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1 + D[0]*theta2 + D[1]*theta4 + D[2]*theta6 + D[3]*theta8)
    scale = np.ones((size[1], size[0]))
    r_0_idx = np.where(r == 0)
    r_non0_idx = np.where(r != 0)
    r[r_0_idx] = 1
    theta_d_r = theta_d / r
    scale[r_non0_idx] = theta_d_r[r_non0_idx]
    u = f[0] * x * scale + c[0]
    v = f[1] * y * scale + c[1]
    tempMatrix[:, :, 0] = u
    tempMatrix[:, :, 1] = v
    return tempMatrix[:, :, 0].astype(np.float32), tempMatrix[:, :, 1].astype(np.float32)

#single fisheye calibration
def get_K_and_D(checkerboard, imgsPath):
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []
    imgpoints = []
    images = glob.glob(imgsPath + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
            #cv2.imshow('findCorners', img)
            #cv2.waitKey(100)
        else:
            print(fname)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, rvecs, tvecs = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    #print("DIM=" + str(_img_shape[::-1]))
    #print("K=np.array(" + str(K.tolist()) + ")")
    #print("D=np.array(" + str(D.tolist()) + ")")

    return DIM, K, D

#dual fisheye stereo calibrate
def get_calibrate_coff(K1, D1, K2, D2, checkerboard, left_images_path, right_images_path, image_size):
    CHECKERBOARD = checkerboard
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objpoints = []
    left_imgpoints = []
    right_imgpoints = []

    left_images = sorted(glob.glob(left_images_path + '/*.jpg'))
    for fname in left_images:
        #print(fname)
        img = cv2.imread(fname)
        #print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            left_imgpoints.append(corners)
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
            #cv2.imshow('www', img)
            #cv2.waitKey(1000)
        else:
            print(fname)

    right_images = sorted(glob.glob(right_images_path + '/*.jpg'))
    for fname in right_images:
        #print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            right_imgpoints.append(corners)
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
        else:
            print(fname)
    print(len(objpoints))

    _, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, K1, D1, K2, D2, image_size,
                                                          flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
                                                          + cv2.fisheye.CALIB_FIX_INTRINSIC,
                                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5))

    print('stereo')
    #print('k1 = ' + str(K1))
    #print('D1 = ' + str(D1))
    #print('K2 = ' + str(K2))
    #print('D2 = ' + str(D2))
    #print('R = ' + str(R))
    #print('T = ' + str(T))
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=0, balance=0.0, fov_scale=1)

    return K1, D1, K2, D2, R1, R2, P1, P2