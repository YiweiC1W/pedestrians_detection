import math
import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt

from imutils.object_detection import non_max_suppression
from _collections import  deque
from centroidtracker import CentroidTracker



ct = CentroidTracker()

#初始化追踪点的列表
trajectory_length = 64
# pts = deque(maxlen=mybuffer)
pts = [deque(maxlen=trajectory_length) for _ in range(1000)]

#定义颜色
cmap = plt.get_cmap('tab20')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, num=20)]


# Histogram of Oriented Gradients Detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# img_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/svm+hog_new/train/STEP-ICCV21-02'
# img_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/SVM+hog+更改中心点/test_img'
# output_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/SVM+hog+更改中心点/output'
img_path = "./train/STEP-ICCV21-02/"
# img_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/SVM+hog/test_img/'
output_path = './svm_output2/'
ii=0
for filename in sorted(os.listdir(img_path)):

    # loop over the image paths
    img = cv2.imread(img_path+'/'+filename)
    orig = img.copy()
     #######可以操作一下########
    # img = imutils.resize(img, width=min(400, img.shape[1]))

    #Resize the frame
    scale_ratio = 0.8
    width = int(img.shape[1] * scale_ratio)
    height = int(img.shape[0] * scale_ratio)
    img = cv2.resize(img, (width, height))


    # img_r = cv2.equalizeHist(img[:, :, 0])
    # img_b = cv2.equalizeHist(img[:, :, 1])
    # img_g = cv2.equalizeHist(img[:, :, 2])
    # img = cv2.merge((img_r, img_b, img_g))
    # print(img.shape)

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
    rects = []

    objects = ct.update1(pick)
    for (objectID, bbx) in objects.items():

        # print("objectID,center")
        # print(objectID, bbx[0], bbx[1], bbx[2], bbx[3])

        Cx = int((bbx[0] + bbx[2]) / 2)
        Cy = int((bbx[1] + bbx[3]) / 2)
        center = (Cx, Cy)
        # print(center)

        # 根据id记录行人行动轨迹
        pts[objectID].append(center)

        # 根据id分配颜色
        color = colors[int(objectID) % len(colors)]
        color = [i * 255 for i in color]

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(img, text, (Cx - 10, Cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), color, 2)
        # cv2.circle(img, (centroid[0], centroid[1]), 4, color, -1)

        # 打印行人行动轨迹
        for j in range(1, len(pts[objectID])):
            if pts[objectID][j - 1] is None or pts[objectID][j] is None:
                continue
            thickness = int(np.sqrt(trajectory_length / float(j + 1)) * 2)
            cv2.line(img, (pts[objectID][j - 1]), pts[objectID][j], color, thickness)

    cv2.imshow('Task1', img)
    cv2.waitKey(1)
    cv2.imwrite(output_path + "/" + filename, img)


    # draw the final bounding boxes
    # for (xA, yA, xB, yB) in pick:
    #     box = [xA, yA, xB, yB]
    #     #将预测的bounding boxes输入
    #     # detections.append([xA, yA, xB, yB])
    #     # detections.append([xA, xB, yA, yB])
    #     rects.append(box)
    #
    #     # draw a bounding box surrounding the object so we can
    #     # visualize it
    #     # (startX, startY, endX, endY) = box
    #     # cv2.rectangle(img, (startX, startY), (endX, endY),
    #     #               (0, 255, 0), 2)
    #
    #     # update our centroid tracker using the computed set of bounding
    #     # box rectangles
    #     objects = ct.update(rects)
    #     # loop over the tracked objects
    #     for (objectID, bbx) in objects.items():
    #
    #
    #         print("objectID,center")
    #         print(objectID,bbx[0],bbx[1],bbx[2],bbx[3])
    #
    #         Cx = int((bbx[0] + bbx[2])/2)
    #         Cy = int((bbx[1] + bbx[3])/2)
    #         center=(Cx,Cy)
    #         print(center)
    #
    #         # 根据id记录行人行动轨迹
    #         pts[objectID].append(center)
    #
    #         # 根据id分配颜色
    #         color = colors[int(objectID) % len(colors)]
    #         color = [i * 255 for i in color]
    #
    #         # draw both the ID of the object and the centroid of the
    #         # object on the output frame
    #         text = "ID {}".format(objectID)
    #         cv2.putText(img, text, (Cx - 10, Cy - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #         cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), color, 2)
    #         # cv2.circle(img, (centroid[0], centroid[1]), 4, color, -1)
    #
    #
    #         # 打印行人行动轨迹
    #         for j in range(1, len(pts[objectID])):
    #             if pts[objectID][j - 1] is None or pts[objectID][j] is None:
    #                 continue
    #             thickness = int(np.sqrt(trajectory_length / float(j + 1)) * 2)
    #             cv2.line(img,(pts[objectID][j-1]),pts[objectID][j],color,thickness)

    # save the output images
    # cv2.imwrite(output_path + "/" + filename, img)
    # cv2.imshow('Task1', img)
    # cv2.waitKey(1)
    # plt.imshow(img)
    # plt.show()
    # print("pts")
    # print(pts)
    #


