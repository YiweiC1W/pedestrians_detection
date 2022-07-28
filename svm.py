import math
import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt

from imutils.object_detection import non_max_suppression
# from collections import  deque
from _collections import  deque
from tracker_svm import *


#初始化追踪点的列表
trajectory_length = 64
# pts = deque(maxlen=mybuffer)
pts = [deque(maxlen=trajectory_length) for _ in range(1000)]

#定义颜色
cmap = plt.get_cmap('tab20')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, num=20)]

# 进行追踪
# 这个函数通过获取同一物体不同时刻的boundingbox的坐标从而实现对其的追踪
Tracker = EuclideanDistTracker()


# Histogram of Oriented Gradients Detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# img_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/svm+hog_new/train/STEP-ICCV21-09'
img_path = "./train/STEP-ICCV21-02/"
# img_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/SVM+hog/test_img/'
output_path = './svm_output/'
# output_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/SVM+hog_new/output/'



for filename in sorted(os.listdir(img_path)):

    # 用于存放boundingbox的起始点坐标、宽、高
    detections = []


    # loop over the image paths
    img = cv2.imread(img_path+'/'+filename)
    orig = img.copy()


    #######可以操作一下########
    img = imutils.resize(img, width=min(800, img.shape[1]))

    #使用Hog人形非类器
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detect people in the image
    (pedestrians, weights) = hog.detectMultiScale(img, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # draw the original bounding boxes

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])
    pick = non_max_suppression(pedestrians, probs=None, overlapThresh=0.5)

    #############################


    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:

        #将预测的bounding boxes输入
        detections.append([xA, yA, xB, yB])
        # detections.append([xA, xB, yA, yB])

        # 物体追踪
        boxer_ids = Tracker.update(detections)  # 同一个物体会有相同的ID
        # print(boxer_ids)
        for box_id in boxer_ids:
            x, y, w, h, id = box_id
            cx = int((x + w) / 2)
            cy = int((y + h) / 2)
            print("circle")
            print((cx,cy))
            center = (cx,cy)

            #根据id记录行人行动轨迹
            pts[id].append(center)

            #根据id分配颜色
            color = colors[int(id)% len(colors)]
            color = [i * 255 for i in color]

            # cv2.putText(img, "Obj" + str(id), (x, y - 15), cv2.FONT_ITALIC, 0.7, (255, 0, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 根据移动物体的轮廓添加boundingbox
            cv2.circle(img, (cx, cy), 5, color, -5)
            cv2.putText(img,"Pedestrians_" + str(id), (x, y - 15), cv2.LINE_AA, 0.6, (255, 255, 255), 2)
            # print(box_id)

            #打印行人行动轨迹
            for j in range(1, len(pts[id])):
                if pts[id][j-1] is None or pts[id][j] is None:
                    continue
                thickness = int(np.sqrt(trajectory_length/float(j+1))*2)
                cv2.line(img,(pts[id][j-1]),pts[id][j],color,thickness)
        cv2.imshow('Task1', img)
        cv2.waitKey(1)
    print("pts")
    print(pts)

    # save the output images
    cv2.imwrite(output_path + "/" + filename, img)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(orig)
    # plt.show()

    print("detections")
    print(detections)

    # #制作视频
    # img_array = []
    # height, width, layers = img.shape
    # size = (width, height)
    # img_array.append(img)
    # out = cv2.VideoWriter('/Users/chenhanxian/PycharmProjects/pythonProject/9517/svm+hog_new/test.avi',cv2.VideoWriter_fourcc(*'DIVX'),25, size)
    #
    #
    # for frame in range(0,len(img_array)):
    #     out.write(img_array[frame])
    #     frame += 1
    # out.release()

