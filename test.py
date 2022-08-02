import os
import time
import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

IMAGE_FOLDER_PATH = "./test/STEP-ICCV21-01/"

test_save="./test"

mask = cv2.imread("mean_mask.jpg")
mask = mask.astype(np.float32)
pics_num=0

def eq(img_eq):
    img_r = cv2.equalizeHist(img_eq[:, :, 0])
    img_b = cv2.equalizeHist(img_eq[:, :, 1])
    img_g = cv2.equalizeHist(img_eq[:, :, 2])
    img_eq = cv2.merge((img_r, img_b, img_g))
    return img_eq

for filename in sorted(os.listdir(IMAGE_FOLDER_PATH)):
    img=cv2.imread(os.path.join(IMAGE_FOLDER_PATH,filename))
    img=img.astype(np.float)

    img1=img.copy()
    img[:, :, 0] -= mask[:,:,0]
    img[:, :, 1] -= mask[:,:,1]
    img[:, :, 2] -= mask[:,:,2]

    used_mask = mask.copy()
    used_mask[:,:,0] -= img1[:, :, 0]
    used_mask[:,:,1] -= img1[:, :, 1]
    used_mask[:,:,2] -= img1[:, :, 2]
    # img[:, :, 0]=mask[:,:,0]-img[:, :, 0]
    # img[:, :, 1]=mask[:,:,1]- img[:, :, 1]
    # img[:, :, 2]=mask[:,:,2]-img[:, :, 2]
    img[img<10]=0
    used_mask[used_mask < 0] = 0
    pics_num+=1
    # img=img.astype(np.uint8)
    img=img[:,:,0]>0
    mask = morphology.remove_small_objects(img, min_size=64)
    plt.imshow(mask, cmap='gray')
    plt.show()



    # img_r = cv2.equalizeHist(img[:, :, 0])
    # img_b = cv2.equalizeHist(img[:, :, 1])
    # img_g = cv2.equalizeHist(img[:, :, 2])
    # img = cv2.merge((img_r, img_b, img_g))


    #
    # used_mask=eq(used_mask)
    #
    # cv2.imwrite(str(pics_num)+".jpg", img)
    # cv2.imshow("mean_mask", img)
    # cv2.waitKey(0)
    # cv2.imshow("mean_mask", used_mask)
    # cv2.waitKey(0)
    # print(pics_num)
