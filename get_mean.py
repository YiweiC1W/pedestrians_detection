import os
import time

import cv2
import numpy as np

output_path = './svm_output1/'
IMAGE_FOLDER_PATH = "./test/STEP-ICCV21-01/"

mean_mask = np.zeros((1080,1920,3))
pics_num=0
for filename in sorted(os.listdir(IMAGE_FOLDER_PATH)):
    img=cv2.imread(os.path.join(IMAGE_FOLDER_PATH,filename))
    mean_mask[:,:,0]=mean_mask[:,:,0]+img[:,:,0]
    mean_mask[:,:,1]= mean_mask[:,:,1]+img[:,:,1]
    mean_mask[:,:,2]=mean_mask[:,:,2]+img[:,:,2]
    pics_num+=1
    print(pics_num)
mean_mask=mean_mask/pics_num
mean_mask=mean_mask.astype(np.uint8)
cv2.imwrite("mean_mask.jpg", mean_mask)

cv2.imshow("mean_mask",mean_mask)
cv2.waitKey(0)
