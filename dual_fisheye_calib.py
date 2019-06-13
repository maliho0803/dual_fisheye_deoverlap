import cv2
import numpy as np
import glob
import scipy.io as sio
from utilitys import camera_calib
_, K1, D1 = camera_calib.get_K_and_D((9, 6), './img/left/')
_, K2, D2 = camera_calib.get_K_and_D((9, 6), './img/right/')

K1, D1, K2, D2, R1, R2, P1, P2 = camera_calib.get_calibrate_coff(K1, D1, K2, D2, (9, 6), './img/left/', './img/right/', (1280, 960))
P2[0, 3] = 0
print('rectify')
print('k1 = ' + str(K1))
print('D1 = ' + str(D1))
print('R1 = ' + str(R1))
print('P1 = ' + str(P1))
print('K2 = ' + str(K2))
print('D2 = ' + str(D2))
print('R2 = ' + str(R2))
print('P2 = ' + str(P2))
save_path = './data/calib_coff.mat'
sio.savemat(save_path, {'K1':K1, 'D1':D1, 'R1':R1, 'P1':P1,'K2':K2, 'D2':D2, 'R2':R2, 'P2':P2,})