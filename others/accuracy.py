import os
import cv2
import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
import matplotlib.pyplot as plt


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


# pridict_path = "outputs_masked"
pridict_path = "result/eq_yolo/test2"
pridect_file = []

label_path = "result\\label2"
label_file = []

is_scale=False

def scale_img(img,scale_ratio):
    width = int(img.shape[1] * scale_ratio)
    height = int(img.shape[0] * scale_ratio)
    img_start = cv2.resize(img, (width, height))
    return img_start

for i in range(6, 400):
    pridect_file.append(os.path.join(pridict_path, str(i) + ".jpg"))
    label_file.append(os.path.join(label_path, str(i) + ".jpg"))


# print(pridect_file)
# print(label_file)
# label_file=get_image_list()


def get_iou(predict, label, i):

    print(f"img{i}")
    predict = predict[:, :, 0]
    label = label[:, :, 0]
    predict[predict != 0] = 255
    label=do_label(label)

    if (is_scale):
        label = scale_img(label,0.8)
    label[label != 0] = 255

    predict_area = np.sum(predict // 255)
    label_area = np.sum(label // 255)
    result = predict * label
    result_area = np.sum(result)

    print(result_area)
    iou=result_area / (predict_area + label_area - result_area)
    print(f"iou={iou}")
    return iou

def do_label(label):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(label)
    stats = stats[1:]
    for x, y, h, w, s in stats:
        label[y:y + w,x:x + h] = 255
    return label
iou=0
for i in range(6,400):
    print(i)
    predict = cv2.imread(pridect_file[i-6])
    label = cv2.imread(label_file[i-6])

    iou+=get_iou(predict,label,i)
print(iou/(400-6))