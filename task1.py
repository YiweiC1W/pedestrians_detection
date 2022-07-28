import math
import cv2
import os
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
from tracker import Tracker
from visualizer import plot_tracker, plot_task2, plot_task3

def hog_clf(descriptor_type = 'default'):
    if descriptor_type == 'daimler':
        # winSize = (48, 96)
        winSize = (48, 96)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    else:
        winSize = (64, 128)
        # winSize = (48, 96)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog


def detect_image(hog, image,i):
    # image = imutils.resize(image, width=min(400, image.shape[1]))
    mask=np.zeros_like(image)
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)
    # (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)


    for(x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)


    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    tracking_info = tracker.update(image, pick, [0.6]*len(pick))
    for bbox in tracking_info:
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        person_id = bbox[4]
        if person_id not in pathss:
            pathss[person_id] = [None] * i
        pathss[person_id].append(bbox)
    result_image = plot_tracker(image, tracking_info, pathss)
    cv2.imwrite("./svm_mask_output/"+str(i)+".jpg", mask)
    # for(xA,yA,xB,yB) in pick:
    #
    #     cv2.rectangle(image, (xA,yA), (xB,yB), (0,255,0), 2)
    return result_image
    return image


def detect_images(hog, images_path):
    i=0
    for image_path in paths.list_images(images_path):
        orig = cv2.imread(image_path)
        image = detect_image(hog, orig,i)
        i+=1
        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
tracker = Tracker(use_cuda=True)
pathss = {}
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
images_path = "./train/STEP-ICCV21-02/"
detect_images(hog, images_path)

