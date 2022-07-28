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
pridict_path = "svm_mask_output"
pridect_file = []

label_path = "train\\0002"
label_file = []
for i in range(10, 31):
    pridect_file.append(os.path.join(pridict_path, str(i) + ".jpg"))
    label_file.append(os.path.join(label_path, "0000" + str(i) + ".png"))


# print(pridect_file)
# print(label_file)
# label_file=get_image_list()


def get_iou(predict, label, i):
    print(f"img{i}")
    predict = predict[:, :, 0]
    label = label[:, :, 0]
    predict[predict != 0] = 255
    label=do_label(label)
    predict_area = np.sum(predict // 255)
    print(predict_area)
    label_area = np.sum(label // 255)
    print(label_area)
    plt.imshow(predict)
    plt.show()
    plt.imshow(label)
    plt.show()
    result = predict * label
    plt.imshow(result)
    plt.show()
    result_area = np.sum(result)

    print(result_area)

    print(f"iou={result_area / (predict_area + label_area - result_area)}")


def do_label(label):
    # label[label != 0] = 255
    #     get_iou(predict,label,i)
    plt.imshow(label)
    plt.show()
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(label)
    stats = stats[1:]
    for x, y, h, w, s in stats:
        label[y:y + w,x:x + h] = 255
    return label
for i in range(10,30):
    print(i)

    predict = cv2.imread(pridect_file[i-10])
    label = cv2.imread(label_file[i-10])
    get_iou(predict,label,i)