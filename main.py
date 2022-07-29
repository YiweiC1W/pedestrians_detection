import argparse
import math
import os
import time

import numpy as np
import torch
import cv2
import sys
sys.path.append(os.getcwd() + '/YOLOX')

from yolox.exp import get_exp


from extractor import Extractor
from tracker import Tracker
from visualizer import plot_tracker, plot_task2, plot_task3

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# IMAGE_FOLDER_PATH = "./train/STEP-ICCV21-02/"
IMAGE_FOLDER_PATH = "./test/STEP-ICCV21-01/"
x1, y1, x2, y2, is_drawing, is_drawing_completed = -1, -1, -1, -1, False, False
original_start_img = None
display_img = None

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def make_parser():
    parser = argparse.ArgumentParser("Pedestrian Detector!")
    parser.add_argument(
        "--path",
        type=str,
        default=IMAGE_FOLDER_PATH,
        help="Path to image folder"
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        help="Task id (1 or 2 or 3)"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default='yolox-s',
        help="model name")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="weights/yolox_s.pth",
        help="path to weights file"
    )
    parser.add_argument("--conf", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--nms", default=0.4, type=float, help="nms threshold")
    parser.add_argument(
        "--video",
        type=bool,
        default=True,
        help="save video"
    )
    parser.add_argument(
        "--picture",
        type=bool,
        default=True,
        help="save picture"
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names



def task_1(args, extractor, tracker, save_folder, files):
    result_files = []

    save_masked_folder = os.path.join(
        'outputs_masked_hist_eq/', args.name + '/', time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    paths = {} # {person_id: [bbox_1, bbox_2, ...]}
    for i, image_name in enumerate(files):

        img = cv2.imread(image_name)
        img_r = cv2.equalizeHist(img[:,:,0])
        img_b = cv2.equalizeHist(img[:,:,1])
        img_g = cv2.equalizeHist(img[:,:,2])
        img = cv2.merge((img_r,img_b,img_g))
        img_filename = os.path.basename(image_name)
        img_info = extractor.extract(img, img_filename)
        tracking_info = tracker.update(img, img_info["bboxes"], img_info["scores"])

        # mask = np.zeros_like(img,dtype=np.uint8)
        # add path for each person
        # bbox[0]: x0, bbox[1]: y0, bbox[2]: x1, bbox[3]: y1
        # bbox[4]: person_id
        for bbox in tracking_info:
            # mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            person_id = bbox[4]
            if person_id not in paths:
                paths[person_id] = [None] * i
            paths[person_id].append(bbox)
        for person_id in paths:
            if len(paths[person_id]) < i + 1:
                paths[person_id].append(None)

        result_image = plot_tracker(img, tracking_info, paths)

        if args.picture or args.video:
            os.makedirs(save_folder, exist_ok=True)
        result_files.append(result_image)
        cv2.namedWindow('Task2')
        cv2.imshow('Task1', result_image)
        cv2.waitKey(1)

        if args.picture:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            print("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

            # save_masked_file_name = os.path.join('outputs_masked', str(i)+".jpg")
            # print("1Saving detection result in {}".format(save_masked_file_name))
            # cv2.imwrite(save_masked_file_name, mask)



    if args.video:
        save_video_name = os.path.join(save_folder, args.name + ".avi")
        print("Saving video in {}".format(save_video_name))
        video_writer = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (result_files[0].shape[1], result_files[0].shape[0]))
        for result_image in result_files:
            video_writer.write(result_image)
        video_writer.release()


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, is_drawing, is_drawing_completed, original_start_img, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            x2 = x
            y2 = y
            display_img = cv2.rectangle(original_start_img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        is_drawing = False
        display_img = cv2.rectangle(original_start_img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)


def is_overlap(bbox1, bbox2):
    if bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1]:
        return False
    return True


def get_distance(bbox1, bbox2):
    mid_x_1 = (bbox1[0] + bbox1[2]) / 2
    mid_x_2 = (bbox2[0] + bbox2[2]) / 2
    mid_y_1 = (bbox1[1] + bbox1[3]) / 2
    mid_y_2 = (bbox2[1] + bbox2[3]) / 2
    return math.sqrt((mid_x_1 - mid_x_2) ** 2 + (mid_y_1 - mid_y_2) ** 2)


def in_group(current_distance_matrix, previous_distance_matrix, boxes, distance_threshold=100):

    result = np.zeros(current_distance_matrix.shape)
    for i in range(len(current_distance_matrix)):
        if len (boxes) and boxes[i] is not None:
            person_1 = boxes[i]
            w1 = person_1[2] - person_1[0]
            h1 = person_1[3] - person_1[1]
        for j in range(len(current_distance_matrix[i])):
            if len(boxes) and boxes[j] is not None and boxes[i] is not None:
                person_2 = boxes[j]
                w2 = person_2[2] - person_2[0]
                h2 = person_2[3] - person_2[1]
                distance_threshold = (w1 + w2) / 1.5
            if current_distance_matrix[i][j] < distance_threshold:
                if i < len(previous_distance_matrix) and j < len(previous_distance_matrix[i]):
                    if previous_distance_matrix[i][j] < distance_threshold and i != j:
                        if abs( h1 - h2 )/h1 < 0.4:
                            result[i][j] = 1

    return result


def task_2(args, extractor, tracker, save_folder, files):
    global x1, y1, x2, y2, is_drawing, is_drawing_completed, original_start_img, display_img
    result_files = []
    unique_person_ids = set()
    person_ids_in = set()
    for i, image_name in enumerate(files):
        img = cv2.imread(image_name)

        print("Processing {}".format(image_name))

        if i == 0:
            original_start_img = img
            display_img = original_start_img.copy()
            instruction = 'Draw a rectangle and press \"Space or Enter\" to continue'
            cv2.putText(img, instruction, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.namedWindow('Task2')
            cv2.setMouseCallback('Task2', draw_rectangle)
            while 1:
                cv2.imshow('Task2', display_img)
                ckey = cv2.waitKey(50)
                if ckey == 13 or ckey == 32:
                    if x1 == -1 or x2 == -1 or y1 == -1 or y2 == -1:
                        continue
                    break

        user_rectangle = (x1, y1, x2, y2)

        img_filename = os.path.basename(image_name)
        img_info = extractor.extract(img, img_filename)
        tracking_info = tracker.update(img, img_info["bboxes"], img_info["scores"])

        person_ids_in.clear()
        for bbox in tracking_info:
            person_id = bbox[4]
            unique_person_ids.add(person_id)
            person_box = (bbox[0], bbox[1], bbox[2], bbox[3])

            if is_overlap(user_rectangle, person_box):
                person_ids_in.add(person_id)

        result_image = plot_task2(img, tracking_info, user_rectangle, unique_person_ids, person_ids_in)
        cv2.imshow('Task2', result_image)
        cv2.waitKey(1)

        if args.picture or args.video:
            os.makedirs(save_folder, exist_ok=True)
        result_files.append(result_image)

        if args.picture:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            print("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

    if args.video:
        save_video_name = os.path.join(save_folder, args.name + ".avi")
        print("Saving video in {}".format(save_video_name))
        video_writer = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                       (result_files[0].shape[1], result_files[0].shape[0]))
        for result_image in result_files:
            video_writer.write(result_image)
        video_writer.release()


def task_3(args, extractor, tracker, save_folder, files):
    result_files = []
    distance_matrices = []
    for i, image_name in enumerate(files):
        img = cv2.imread(image_name)

        print("Processing {}".format(image_name))

        img_filename = os.path.basename(image_name)
        img_info = extractor.extract(img, img_filename)
        tracking_info = tracker.update(img, img_info["bboxes"], img_info["scores"])

        groups = [] # [set(person_ids), set(person_ids), ...]
        person_ids = [bbox[4] for bbox in tracking_info]
        result_image = img.copy()
        if len(tracking_info):
            boxes = [None] * (max(person_ids) + 1)
            distance_matrix = np.full((max(person_ids) + 1, max(person_ids) + 1), np.inf)
            for z, bbox in enumerate(tracking_info):
                person_id = bbox[4]
                person_box = (bbox[0], bbox[1], bbox[2], bbox[3])
                boxes[person_id] = person_box
                for j, bbox2 in enumerate(tracking_info):
                    person_id2 = bbox2[4]
                    person_box2 = (bbox2[0], bbox2[1], bbox2[2], bbox2[3])
                    dist = get_distance(person_box, person_box2)
                    distance_matrix[person_id][person_id2] = dist
                    distance_matrix[person_id2][person_id] = dist
            distance_matrices.append(distance_matrix)

            if i > 0:
                last_frame_distance_matrix = distance_matrices[-1]
                connectivity_matrix = in_group(distance_matrix, last_frame_distance_matrix, boxes)
                for person_id_i in range(1, len(connectivity_matrix)):
                    for person_id_j in range(1, len(connectivity_matrix[person_id_i])):
                        if connectivity_matrix[person_id_i][person_id_j] == 1:
                            already_in_group = False
                            for group in groups:
                                if person_id_i in group:
                                    group.add(person_id_j)
                                    already_in_group = True
                                    break
                                if person_id_j in group:
                                    group.add(person_id_i)
                                    already_in_group = True
                                    break
                            if not already_in_group:
                                groups.append({person_id_i, person_id_j})

                id_in_group = set()
                for group in groups:
                    for person_id in group:
                        id_in_group.add(person_id)
                nbs_not_in_group = len(set(person_ids)) - len(id_in_group)
                result_image = plot_task3(img, boxes, groups, len(id_in_group), nbs_not_in_group)


        cv2.namedWindow('Task3')
        cv2.imshow('Task3', result_image)
        cv2.waitKey(1)

        if args.picture or args.video:
            os.makedirs(save_folder, exist_ok=True)
        result_files.append(result_image)

        if args.picture:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            print("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

    if args.video:
        save_video_name = os.path.join(save_folder, args.name + ".avi")
        print("Saving video in {}".format(save_video_name))
        video_writer = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                       (result_files[0].shape[1], result_files[0].shape[0]))
        for result_image in result_files:
            video_writer.write(result_image)
        video_writer.release()




if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(exp_name=args.name)

    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
    model.eval()

    weights_file = args.weights

    print("Loading weights from {}".format(weights_file))
    weights = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(weights["model"])
    print("Loaded weights from {}".format(weights_file))

    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    current_time = time.localtime()

    # Initialize the yolox as our feature extractor
    extractor = Extractor(args, model, exp)

    # Initialize the deep sort tracker
    tracker = Tracker(args.device == "gpu")

    save_folder = os.path.join(
        'outputs_eq/', args.name + '/', time.strftime("%Y_%m_%d_%H_%M_%S", current_time) + 'task_' + str(args.task)
    )

    if args.task == 1:
        task_1(args, extractor, tracker, save_folder, files)
    elif args.task == 2:
        task_2(args, extractor, tracker, save_folder, files)
    elif args.task == 3:
        task_3(args, extractor, tracker, save_folder, files)
    cv2.destroyAllWindows()
