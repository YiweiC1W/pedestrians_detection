import math
import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt
from skimage import morphology
from imutils.object_detection import non_max_suppression
from _collections import  deque
from centroidtracker import CentroidTracker


IMAGE_FOLDER_PATH = "./test/STEP-ICCV21-01/"

test_save="./df1"

# img1 = cv2.imread(os.path.join(IMAGE_FOLDER_PATH,"000062.jpg"))
# img2 = cv2.imread(os.path.join(IMAGE_FOLDER_PATH,"000059.jpg"))
# img1=img1.astype(np.float)
# img2=img2.astype(np.float)
#
# mask= np.zeros_like(img1)
# imgs=sorted(os.listdir(IMAGE_FOLDER_PATH))
# for file_num in range(2,len(imgs)):
#     img_later = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, imgs[file_num]))
#     img_start = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, imgs[file_num-2]))
#     img_later = img_later.astype(np.float)
#     img_start = img_start.astype(np.float)
#
#     for i in range(3):
#         mask[:,:,i]=img_later[:,:,i]-img_start[:,:,i]
#     mask[mask<0]=0
#     mask=mask[:,:,0]
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#     (pedestrians, weights) = hog.detectMultiScale(mask, winStride=(4, 4),
#                                                   padding=(8, 8), scale=1.05)
#
#     plt.imshow(mask[:,:,0],cmap='gray')
#     plt.show()
#     # cv2.imshow("1",mask)
#     # cv2.waitKey(0)
#     cv2.imwrite(os.path.join(test_save,str(file_num)+".jpg"),mask)
#
#




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
# # output_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/SVM+hog+更改中心点/output'
# img_path = "./train/STEP-ICCV21-02/"
# # img_path = '/Users/chenhanxian/PycharmProjects/pythonProject/9517/SVM+hog/test_img/'
# output_path = './svm_output2/'
ii=0

imgs=sorted(os.listdir(IMAGE_FOLDER_PATH))

for file_num in range(10,len(imgs)):
    filename=imgs[file_num]
    mask = np.zeros((1080,1920))
    img_later = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, imgs[file_num]))
    img_start = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, imgs[file_num - 10]))
    img_later = img_later.astype(np.float)
    img_start = img_start.astype(np.float)

    mask = img_later[:, :, 0] - img_start[:, :, 0]
    # plt.imsave(os.path.join(test_save,str(file_num)+"-0.jpg"),mask,cmap='gray')
    mask=np.absolute(mask)

    mask[mask < 15] = 0
    mask=mask.astype(np.uint8)
    plt.imsave(os.path.join(test_save,str(file_num)+"-1.jpg"),mask,cmap='gray')

    mask=mask>0
    plt.imsave(os.path.join(test_save,str(file_num)+"-1-1.jpg"),mask,cmap='gray')

    # plt.imshow(mask, cmap='gray')
    # plt.show()


    # cv2.imshow('Task1', mask)
    # cv2.waitKey(1)
    # orig = img.copy()
     #######可以操作一下########
    # img = imutils.resize(img, width=min(400, img.shape[1]))

    #Resize the frame, connectivity=1
    mask = morphology.remove_small_objects(mask, min_size=512)
    plt.imsave(os.path.join(test_save,str(file_num)+"-2.jpg"),mask,cmap='gray')

    k=np.ones((4*4),np.uint8)
    mask=mask.astype(np.uint8)
    mask=cv2.dilate(mask,k,1)
    plt.imsave(os.path.join(test_save,str(file_num)+"-3.jpg"),mask,cmap='gray')

    # mask[mask == True] = 1
    # mask[mask == False] = 0

    img_start[:,:,0]*=mask
    img_start[:,:,1]*=mask
    img_start[:,:,2]*=mask
    # scale_ratio = 0.8
    # width = int(img_start.shape[1] * scale_ratio)
    # height = int(img_start.shape[0] * scale_ratio)
    # img_start = cv2.resize(img_start, (width, height))

    # mask = morphology.remove_small_holes(mask, area_threshold=5, connectivity=1)

    # mask=cv2.equalizeHist(mask)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    img_start=img_start.astype(np.uint8)
    # cv2.imshow('Task1', img_start)
    # cv2.waitKey(1)
    img_r = cv2.equalizeHist(img_start[:, :, 0])
    img_b = cv2.equalizeHist(img_start[:, :, 1])
    img_g = cv2.equalizeHist(img_start[:, :, 2])
    img_start = cv2.merge((img_r, img_b, img_g))
    # print(img.shape)
    cv2.imwrite(os.path.join(test_save,str(file_num)+".jpg"),img_start)
    #使用Hog人形非类器
    # hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #
    # # detect people in the image
    # (pedestrians, weights) = hog.detectMultiScale(img_start, winStride=(4, 4),
    #     padding=(8, 8), scale=1.05)
    #
    # # draw the original bounding boxes
    # # apply non-maxima suppression to the bounding boxes using a
    # # fairly large overlap threshold to try to maintain overlapping
    # # boxes that are still people
    # pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])
    # pick = non_max_suppression(pedestrians, probs=None, overlapThresh=0.5)
    #
    # #############################
    # rects = []
    #
    # objects = ct.update1(pick)
    # for (objectID, bbx) in objects.items():
    #
    #     # print("objectID,center")
    #     # print(objectID, bbx[0], bbx[1], bbx[2], bbx[3])
    #
    #     Cx = int((bbx[0] + bbx[2]) / 2)
    #     Cy = int((bbx[1] + bbx[3]) / 2)
    #     center = (Cx, Cy)
    #     # print(center)
    #
    #     # 根据id记录行人行动轨迹
    #     pts[objectID].append(center)
    #
    #     # 根据id分配颜色
    #     color = colors[int(objectID) % len(colors)]
    #     color = [i * 255 for i in color]
    #
    #     # draw both the ID of the object and the centroid of the
    #     # object on the output frame
    #     text = "ID {}".format(objectID)
    #     cv2.putText(img_start, text, (Cx - 10, Cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #     cv2.rectangle(img_start, (bbx[0], bbx[1]), (bbx[2], bbx[3]), color, 2)
    #     # cv2.circle(img, (centroid[0], centroid[1]), 4, color, -1)
    #
    #     # 打印行人行动轨迹
    #     # for j in range(1, len(pts[objectID])):
    #     #     if pts[objectID][j - 1] is None or pts[objectID][j] is None:
    #     #         continue
    #     #     thickness = int(np.sqrt(trajectory_length / float(j + 1)) * 2)
    #     #     cv2.line(img_start, (pts[objectID][j - 1]), pts[objectID][j], color, thickness)
    #
    # cv2.imshow('Task1', img_start)
    # cv2.waitKey(1)
    # cv2.imwrite(output_path + "/" + filename, img)

