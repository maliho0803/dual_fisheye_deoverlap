import cv2
import tensorflow as tf
import scipy.io as sio
import numpy as np
from utilitys import camera_calib
from utilitys import prediction
from utilitys import disparity_core
import glob

#get calib params
calib_param = sio.loadmat('./data/calib_coff.mat')
K1 = calib_param['K1']
D1 = calib_param['D1']
K2 = calib_param['K2']
D2 = calib_param['D2']
R1 = calib_param['R1']
R2 = calib_param['R2']
P1 = calib_param['P1']
P2 = calib_param['P2']

pic_path_list = glob.glob('./img/*.jpg')
PATH_TO_CKPT = './data/frozen_inference_graph_yunqi_160000_640.pb'

label_dict = {
              "0": {"weight":-1,"chinese_name":"空盒子","name":"box","online_code": -1},
              "1": {"weight":554.8,"chinese_name":"统一冰红茶柠檬","name":"TongYiBingHongChaNingMeng","online_code": 1003733},
              "2": {"weight":538.8,"chinese_name":"统一绿茶茉莉","name":"TongYiLvChaMoLi","online_code": 1003740},
              "3": {"weight":498.5,"chinese_name":"雅哈冰咖啡","name":"YaHaBingKaFei","online_code": 1003734},
              "4": {"weight":549.6,"chinese_name":"小茗同学青柠红茶","name":"XiaoMingTongXueQingNingHongCha","online_code": 1003742},
              "5": {"weight":500.7,"chinese_name":"统一鲜橙多","name":"TongYiXianChengDuo","online_code": 1003735},
              "6": {"weight":116.2,"chinese_name":"汤达人日式豚骨拉面杯","name":"TangDaRenRiShiTunGuLaMianBei","online_code": 1003737},
              "7": {"weight":123.2,"chinese_name":"汤达人酸酸辣辣豚骨拉面杯","name":"TangDaRenSuanSuanLaLaTunGuLaMianBei","online_code": 1003736},
              "8": {"weight":551,"chinese_name":"统一阿萨姆奶茶原味","name":"TongYiASaMuNaiChaYuanWei","online_code": 1003733},
              "9": {"weight":312.5,"chinese_name":"雅哈意式经典","name":"YaHaYiShiJingDian","online_code": 1003740},
              "10": {"weight":547.8,"chinese_name":"小茗同学溜溜哒茶","name":"XiaoMingTongXueLiuLiuDaCha","online_code": 1003734},
              "11": {"weight":271.3,"chinese_name":"统一奶茶草莓","name":"TongYiNaiChaCaoMei","online_code": 1003742},
              "12": {"weight":270.6,"chinese_name":"统一奶茶巧克力","name":"TongYiNaiChaQiaoKeLi","online_code": 1003742},
              "13": {"weight":270.1,"chinese_name":"统一奶茶麦香","name":"TongYiNaiChaMaiXiang","online_code": 1003737},
              "14": {"weight":449.6,"chinese_name":"茶瞬鲜柠檬绿茶","name":"ChaShunXianNingMengLvCha","online_code": 1003736},
              "15": {"weight":285,"chinese_name":"REMIX爱混牛乳布丁奶茶","name":"REMIXAiHunNiuRuBuDingNaiCha","online_code": 1003736},
              "16": {"weight":449.4,"chinese_name":"茶瞬鲜青桔乌龙茶","name":"ChaShunXianQingJuWuLongCha","online_code": 1003736},
              "17": {"weight":449.4,"chinese_name":"统一爱夸饮用矿泉水","name":"ALKAQUA","online_code": 1007105}
            }

#de-fishey, stereo-rectify
test_left = cv2.imread('./img/test.0.206.jpg')
test_right = cv2.imread('./img/test.1.206.jpg')

lmap1, lmap2 = camera_calib.get_distort_map(K1, D1, R1, P1, size=(1280, 960), alpha=0.0, fov_scale=1)
rmap1, rmap2 = camera_calib.get_distort_map(K2, D2, R2, P2, size=(1280, 960), alpha=0.0, fov_scale=1)

res1 = cv2.remap(test_left, lmap1, lmap2, cv2.INTER_CUBIC)
res2 = cv2.remap(test_right, rmap1, rmap2, cv2.INTER_CUBIC)

res1 = cv2.resize(res1, (640, 480))
res2 = cv2.resize(res2, (640, 480))
cv2.imwrite('./result/ltest.jpg', res1)
cv2.imwrite('./result/rtest.jpg', res2)
#img_show1 = np.hstack((res1, res2))
#cv2.imshow('img_stack', img_show1)
#cv2.imshow('res1', res1)
#cv2.imshow('res2', res2)
#cv2.waitKey(0)

#calculate disparity
window_size = 1
min_disp = 0
max_disparity = 128 - min_disp
stereoProcessor = cv2.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=max_disparity,
                              blockSize=3,
                              P1=8 * 3 * window_size ** 2,
                              P2=32 * 3 * window_size ** 2,
                              disp12MaxDiff=1,
                              uniquenessRatio=10,
                              speckleWindowSize=100,
                              speckleRange=32)

grayL = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

disparity = stereoProcessor.compute(grayL, grayR)

_, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
disparity_scaled = (disparity / 16.).astype(np.uint8)

sio.savemat('./result/disparity.mat', {'disparity':disparity_scaled})
cv2.imwrite('./result/disparity.jpg', (disparity_scaled * (256. / max_disparity)).astype(np.uint8))
#cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8))
#cv2.waitKey(0)

#get object detection results
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session()
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    image_np_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    imgL = cv2.resize(res1, (600, 600))
    imgR = cv2.resize(res2, (600, 600))
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    confidence_threshold = 0.5

    #detect left and right image
    detection_classes, detection_boxes, detection_scores = prediction.get_detect_result([imgL, imgR],
                                                                                        confidence_threshold,
                                                                                        sess,
                                                                                        tensor_dict,
                                                                                        image_np_tensor)
detection_boxesL = detection_boxes[0]
detection_classesL = detection_classes[0]
detection_scoresL = detection_scores[0]
detection_boxesR = detection_boxes[1]
detection_classesR = detection_classes[1]
detection_scoresR = detection_scores[1]
h, w, c = res1.shape
print(h,w)

#de-overlap processing
num_boxes_L = len(detection_boxesL)
num_boxes_R = len(detection_boxesR)

#save detect result
for m in range(num_boxes_L):
    Lbox = detection_boxesL[m]
    x1 = int(w * Lbox[2])
    x2 = int(w * Lbox[3])
    y1 = int(h * Lbox[0])
    y2 = int(h * Lbox[1])
    Lbox_label = detection_classesL[m]
    Lbox_scores = detection_scoresL[m]
    det_imgL = cv2.rectangle(res1, (x1, y1), (x2, y2), (255, 0, 0), 1)
    det_imgL = cv2.putText(det_imgL, str(Lbox_label) + '_' + str(int(Lbox_scores * 100)),
                           (x1, y1 - 5), 1, 1, (0, 255, 255), 1)
    # det_imgL = cv2.cvtColor(det_imgL, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./result/det_imgL.jpg', det_imgL)

for n in range(num_boxes_R):
    Rbox = detection_boxesR[n]
    x1 = int(w * Rbox[2])
    x2 = int(w * Rbox[3])
    y1 = int(h * Rbox[0])
    y2 = int(h * Rbox[1])
    Rbox_label = detection_classesR[n]
    Rbox_scores = detection_scoresR[n]
    det_imgR = cv2.rectangle(res2, (x1, y1), (x2, y2), (255, 0, 0), 1)
    det_imgR = cv2.putText(det_imgR, str(Rbox_label) + '_' + str(int(Rbox_scores * 100)),
                           (x1, y1 - 5), 1, 1, (0, 255, 255), 1)
    # det_imgR = cv2.cvtColor(det_imgR, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./result/det_imgR.jpg', det_imgR)

# final_boxes = detection_boxesL
# final_classes = detection_classesL
# final_scores = detection_scoresL
print(detection_classesL)
print(detection_classesR)
overlap_index = []

for i in range(num_boxes_L):
    Lbox = detection_boxesL[i]
    Lbox_label = detection_classesL[i]
    Lcenter_x = int((Lbox[2] + Lbox[3]) * w / 2)
    Lcenter_y = int((Lbox[0] + Lbox[1]) * h / 2)
    y1 = int(h * Lbox[0])
    y2 = int(h * Lbox[1])
    x1 = int(w * Lbox[2])
    x2 = int(w * Lbox[3])
    print('left_center=' + str(Lcenter_y) + ',' + str(Lcenter_x))
    print('left_labels=' + str(Lbox_label))

    # calculate average disparity in bounding box
    avg_disparity = disparity_core.construct_disparity_hist(disparity_scaled, (y1, y2, x1, x2))
    avg_disparity = int(avg_disparity)
    print('avg_disparity=' + str(avg_disparity))

    for j in range(num_boxes_R):
        Rbox = detection_boxesR[j]
        Rbox_label = detection_classesR[j]
        Rcenter_x = int((Rbox[2] + Rbox[3]) * w / 2)
        Rcenter_y = int((Rbox[0] + Rbox[1]) * h / 2)
        print('right_center=' + str(Rcenter_y) + ',' + str(Rcenter_x))
        print('right_labels=' + str(Rbox_label))
        if abs(Lcenter_x - avg_disparity - Rcenter_x) < 16 \
                and abs(Lcenter_y - Rcenter_y) < 16 and Lbox_label == Rbox_label:
            print('overlapped, deleted it')
            overlap_index.append(j)

print(overlap_index)
# for index in overlap_index:
#     del detection_scoresR[index]
#     del detection_classesR[index]
#     del detection_boxesR[index]
