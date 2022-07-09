import argparse
import os
import time

import torch
import cv2
from yolox.exp import get_exp
from yolox.data import COCO_CLASSES
from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
IMAGE_FOLDER_PATH = "//mnt/a/OneDrive/UNSW/COMP9517/Group_Project/train/STEP-ICCV21-02/"


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
    parser.add_argument("--conf", default=0.3, type=float, help="confidence threshold")
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


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names

        self.num_classes = exp.num_classes
        self.confthre = args.conf
        self.nmsthre = args.nms
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)

            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            print("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = visualize(img, bboxes, scores, cls, cls_conf)
        return vis_res


    def get_person_list(self, output, img_info, conf=0.35):
        img = img_info["raw_img"]
        ratio = img_info["ratio"]
        filename = img_info["file_name"]

        if output is None:
            return img

        output = output.cpu()
        boxes = output[:, 0:4]

        # preprocessing: resize
        boxes /= ratio
        cls_ids = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        person_list_tensor = []
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            if cls_id != 0:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            person = {
                "filename": filename,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "score": score,
                "mid_x": (x0 + x1) // 2,
                "mid_y": (y0 + y1) // 2,
                "mid_xy": ((x0 + x1) // 2, (y0 + y1) // 2)
            }
            person_list_tensor.append(person)
        return person_list_tensor


def visualize(img, boxes, scores, cls_ids, conf=0.5):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        if cls_id != 0:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)
    return img


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
    file_name = os.path.join(exp.output_dir, exp.exp_name)
    os.makedirs(file_name, exist_ok=True)

    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
    model.eval()

    weights_file = args.weights

    print("Loading weights from {}".format(weights_file))
    weights = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(weights["model"])
    print("Loaded weights from {}".format(weights_file))

    predictor = Predictor(
        model, exp, COCO_CLASSES,
        args.device,
    )

    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    current_time = time.localtime()
    save_folder = os.path.join(
        'outputs/', args.name + '/', time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    result_files = []
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

        os.makedirs(save_folder, exist_ok=True)

        person_list = predictor.get_person_list(outputs[0], img_info, predictor.confthre)

        if args.person:
            save_json_name = os.path.join(save_folder, os.path.basename(image_name).split('.')[0] + "_person_list.txt")
            print("Saving person list to {}".format(save_json_name))
            with open(save_json_name, "w") as f:
                for person in person_list:
                    f.write(str(person) + "\n")

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
