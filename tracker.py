from deep_sort import DeepSort
import torch

config = {
    'MODEL': './weights/ckpt.t7',
    'MAX_DIST': 0.2,
    'MIN_CONFIDENCE': 0.3,
    'NMS_MAX_OVERLAP': 1,
    'MAX_IOU_DISTANCE': 0.7,
    'MAX_AGE': 70,
    'N_INIT': 3,
    'NN_BUDGET': 120,
}


def fix_box(boxes):
    for i in range(len(boxes)):
        box = boxes[i]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        x0 = int(x0 + w / 2)
        y0 = int(y0 + h / 2)
        x1 = int(x1 + w / 2)
        y1 = int(y1 + h / 2)
        box[0] = x0
        box[1] = y0
        box[2] = x1
        box[3] = y1
    return boxes


class Tracker(object):
    def __init__(self, use_cuda=False):
        self.deep_sort = DeepSort(
            model_path= config['MODEL'],
            max_dist= config['MAX_DIST'],
            min_confidence= config['MIN_CONFIDENCE'],
            nms_max_overlap= config['NMS_MAX_OVERLAP'],
            max_iou_distance= config['MAX_IOU_DISTANCE'],
            max_age= config['MAX_AGE'],
            n_init= config['N_INIT'],
            nn_budget= config['NN_BUDGET'],
            use_cuda = use_cuda
        )

    def update(self, img, bboxs, scores):
        if len(bboxs) == 0:
            return []

        bboxs_xywh = []
        for bbox in bboxs:
            bboxs_xywh.append([bbox[0], bbox[1], abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])])
        bboxs_xywh = torch.Tensor(bboxs_xywh)
        outputs = self.deep_sort.update(bboxs_xywh, scores, img)
        return fix_box(outputs)

