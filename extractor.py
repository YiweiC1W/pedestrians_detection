import os

from yolox.data import COCO_CLASSES
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
import cv2
import torch





class Extractor(object):
    def __init__(
            self,
            args,
            model,
            exp,
    ):
        self.model = model
        self.cls_names = COCO_CLASSES,
        self.num_classes = exp.num_classes
        self.confthre = args.conf
        self.nmsthre = args.nms
        self.test_size = exp.test_size
        self.device = args.device
        self.preproc = ValTransform(legacy=False)


    def extract(self, img, filename):
        img_info = {"file_name": filename}

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
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

        img_info['bboxes'], img_info['scores'], img_info['cls_ids'], img_info['box_nums'] = [], [], [], 0

        outputs = outputs[0]
        bboxes = outputs[:, 0:4] / ratio
        scores = outputs[:, 4] * outputs[:, 5]
        cls_ids = outputs[:, 6]

        if outputs[0] is None:
            img_info['bboxes'], img_info['scores'], img_info['cls_ids'], img_info['box_nums'] = [], [], [], 0
            return img_info

        for i, value in enumerate(cls_ids):
            if value == 0:
                img_info['cls_ids'].append(value)
                img_info['scores'].append(scores[i])
                img_info['bboxes'].append(bboxes[i])
                img_info['box_nums'] += 1

        return img_info
