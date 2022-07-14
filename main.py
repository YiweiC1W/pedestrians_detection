import argparse
import os
import time
import torch
import cv2
import sys
sys.path.append(os.getcwd() + '/YOLOX')

from yolox.exp import get_exp


from extractor import Extractor
from tracker import Tracker
from visualizer import plot_tracker, plot_task2

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
IMAGE_FOLDER_PATH = "./train/STEP-ICCV21-09/"
x1, y1, x2, y2, is_drawing, is_drawing_completed = -1, -1, -1, -1, False, False
original_start_img = None
display_img = None


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

    paths = {} # {person_id: [bbox_1, bbox_2, ...]}
    for i, image_name in enumerate(files):

        img = cv2.imread(image_name)
        img_filename = os.path.basename(image_name)
        img_info = extractor.extract(img, img_filename)
        tracking_info = tracker.update(img, img_info["bboxes"], img_info["scores"])

        # add path for each person
        # bbox[0]: x0, bbox[1]: y0, bbox[2]: x1, bbox[3]: y1
        # bbox[4]: person_id
        for bbox in tracking_info:
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
        key = cv2.waitKey(1)  # delay 200ms

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
    pass



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
        'outputs/', args.name + '/', time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )


    if args.task == 1:
        task_1(args, extractor, tracker, save_folder, files)
    elif args.task == 2:
        task_2(args, extractor, tracker, save_folder, files)
    elif args.task == 3:
        task_3(args, extractor, tracker, save_folder, files)
    cv2.destroyAllWindows()
