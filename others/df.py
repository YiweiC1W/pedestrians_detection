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


IMAGE_FOLDER_PATH = "../9517dataset/test/STEP-ICCV21-01/"

test_save="../df1"
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
df_num=10
# mask = morphology.remove_small_holes(mask, area_threshold=5, connectivity=1)


for file_num in range(df_num,len(imgs)):
    filename=imgs[file_num]
    mask = np.zeros((1080,1920))
    img_later = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, imgs[file_num]))
    img_start = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, imgs[file_num - df_num]))
    img_later = img_later.astype(np.float)
    img_start = img_start.astype(np.float)

    mask = img_later[:, :, 0] - img_start[:, :, 0]
    # plt.imsave(os.path.join(test_save,str(file_num)+"-0.jpg"),mask,cmap='gray')
    mask=np.absolute(mask)

    # Filter out pixels with small differences
    mask[mask < 15] = 0
    mask=mask.astype(np.uint8)
    plt.imsave(os.path.join(test_save,str(file_num)+"-1.jpg"),mask,cmap='gray')

    # binarize img
    mask=mask>0
    plt.imsave(os.path.join(test_save,str(file_num)+"-1-1.jpg"),mask,cmap='gray')

    # remove small connecttes area
    mask = morphology.remove_small_objects(mask, min_size=256)
    plt.imsave(os.path.join(test_save,str(file_num)+"-2.jpg"),mask,cmap='gray')

    #dialte
    k=np.ones((4*4),np.uint8)
    mask=mask.astype(np.uint8)
    mask=cv2.dilate(mask,k,1)
    plt.imsave(os.path.join(test_save,str(file_num)+"-3.jpg"),mask,cmap='gray')

    #used as mask
    img_start[:,:,0]*=mask
    img_start[:,:,1]*=mask
    img_start[:,:,2]*=mask

    if (is_scale):
        img = scale_img(img_start,0.8)
    img_start=img_start.astype(np.uint8)

    cv2.imwrite(os.path.join(test_save,str(file_num)+".jpg"),img_start)


