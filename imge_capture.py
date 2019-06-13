# -*- coding: utf-8 -*-
import cv2

camera_list = [0, 1]
start_count = 207
caps = {}
cameras = {s: camera_list[s] for s in range(2)}
for i, c in cameras.items():

    cap = cv2.VideoCapture(c)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    caps[i] = cap
    ret, _ = cap.read()
    print(i, ret)

while True:
    for i, cap in caps.items():
        if cap.grab():
            _, frame = cap.retrieve()
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow(str(i), frame)
            #cv2.waitKey(30)
    c = cv2.waitKey(1)
    if c == 32 or c == 13:
        for i, cap in caps.items():
            _, frame = cap.read()
            filename = '.'.join(['test', str(i), str(start_count), 'jpg'])
            print(filename)
            cv2.imwrite(filename, frame)
        start_count += 1
        print("now count is: ", start_count)


