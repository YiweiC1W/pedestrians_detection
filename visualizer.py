import cv2
import numpy as np
from YOLOX.yolox.utils.visualize import _COLORS


def plot_tracker(img, boxes, paths):
    for i in range(len(boxes)):
        box = boxes[i]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        person_id = box[4]

        color_ = _COLORS[person_id % _COLORS.shape[0]]
        color = (color_ * 255).astype(np.uint8).tolist()

        path_history = paths[person_id]
        for path in path_history:
            if path is not None:
                mid_x = int((path[0] + path[2]) / 2)
                mid_y = int((path[1] + path[3]) / 2)
                cv2.circle(img, (mid_x, mid_y), 2, color, -1)


        text = '%d' % (person_id)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (color_ * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )

        # draw tracking point (midpoint of the box)
        # img[mid_x][mid_y] = color

        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
