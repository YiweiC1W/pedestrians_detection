import argparse
import os
import time

import torch
import cv2
from yolox.exp import get_exp


from extractor import Extractor
from tracker import Tracker
from visualizer import plot_tracker

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
IMAGE_FOLDER_PATH = "./train/STEP-ICCV21-02/"


def make_parser():
    parser = argparse.ArgumentParser("Pedestrian Detector!")
    parser.add_argument(
        "--path",
        type=str,
        default=IMAGE_FOLDER_PATH,
        help="Path to image folder"
    )
    parser.add_argument(
        "--device",
        default="cpu",
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
    parser.add_argument("--nms", default=0.3, type=float, help="nms threshold")
    parser.add_argument(
        "--video",
        type=bool,
        default=True,
        help="save video"
    )
    parser.add_argument(
        "--person",
        type=bool,
        default=True,
        help="save person list"
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



def main(exp, args):
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
    tracker = Tracker()

    save_folder = os.path.join(
        'outputs/', args.name + '/', time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
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

        os.makedirs(save_folder, exist_ok=True)

        if args.picture:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            print("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
            result_files.append(result_image)

    if args.video:
        save_video_name = os.path.join(save_folder, args.name + ".avi")
        print("Saving video in {}".format(save_video_name))
        video_writer = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (result_files[0].shape[1], result_files[0].shape[0]))
        for result_image in result_files:
            video_writer.write(result_image)
        video_writer.release()



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(exp_name=args.name)
    main(exp, args)
