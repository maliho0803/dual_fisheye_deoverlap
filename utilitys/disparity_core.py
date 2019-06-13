import numpy
import cv2
#构建视差直方图，32bin
def construct_disparity_hist(disparity, box=[0, 0 , 0, 0]):
    count_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sum_bin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y1 = box[0]
    y2 = box[1]
    x1 = box[2]
    x2 = box[3]
    for i in range(y1, y2):
        for j in range(x1, x2):
            if disparity[i][j] == 0:
                continue
            remain = disparity[i][j] // 20
            count_hist[remain] = count_hist[remain] + 1
            sum_bin[remain] = sum_bin[remain] + disparity[i][j]

    max_index = count_hist.index(max(count_hist))
    print(count_hist)
    print(sum_bin)
    return sum_bin[max_index] / count_hist[max_index]




def calc_avg_disparity(disparity, box=(0, 0, 0, 0)):
    '''box: y1, y2, x1, x2'''
    y1 = box[0]
    y2 = box[1]
    x1 = box[2]
    x2 = box[3]
    valid_disparity = 0
    sum_disparity = 0
    for i in range(y1, y2):
        for j in range(x1, x2):
            if disparity[i][j] == 0:
                continue
            sum_disparity += disparity[i][j]
            valid_disparity = valid_disparity + 1

    avg_disparity = sum_disparity / valid_disparity
    return (avg_disparity)