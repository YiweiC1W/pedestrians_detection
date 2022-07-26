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
        for i, path in enumerate(path_history):
            if path is not None:
                current_x = int((path[0] + path[2]) / 2)
                current_y = int((path[1] + path[3]) / 2)

                for j in range(i, 0, -1):
                    if path_history[j] is not None:
                        previous_x = int((path_history[j][0] + path_history[j][2]) / 2)
                        previous_y = int((path_history[j][1] + path_history[j][3]) / 2)
                        cv2.line(img, (current_x, current_y), (previous_x, previous_y), color, 2)
                        current_x = previous_x
                        current_y = previous_y


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


def plot_task2(img, boxes, user_rectangle, unique_ids, nbs_in):
    count_of_unique_ids = 'Unique pedestrians detected: ' + str(len(unique_ids))
    count_of_current_person = 'Current person: ' + str(len(boxes))
    count_of_nbs_in = 'Within region: ' + str(len(nbs_in))
    txt_color = (255, 255, 255)

    cv2.putText(img, count_of_unique_ids, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)
    cv2.putText(img, count_of_current_person, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)
    cv2.putText(img, count_of_nbs_in, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)
    cv2.rectangle(img, (user_rectangle[0], user_rectangle[1]), (user_rectangle[2], user_rectangle[3]), (0, 0, 255), 2)

    for i in range(len(boxes)):
        box = boxes[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        cv2.rectangle(img, (x0, y0), (x1, y1), color=(10,250,0), thickness=2)

    return img

def plot_task3(img, boxes, groups, forming_groups, leaving_person_ids, leave_ids, enter_ids,  last_boxes, nbs_in_groups, nbs_not_in_groups):
    nbs_in_groups = 'Within group: ' + str(nbs_in_groups)
    nbs_not_in_groups = 'Not in group: ' + str(nbs_not_in_groups)
    for i, group in enumerate(groups):
        person = group.pop()
        group_x0 = boxes[person][0]
        group_y0 = boxes[person][1]
        group_x1 = boxes[person][2]
        group_y1 = boxes[person][3]

        for person_id in group:
            box = boxes[person_id]
            if box is None:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            if x0 < group_x0:
                group_x0 = x0
            if y0 < group_y0:
                group_y0 = y0
            if x1 > group_x1:
                group_x1 = x1
            if y1 > group_y1:
                group_y1 = y1

        color_ = _COLORS[i % _COLORS.shape[0]]
        color = (color_ * 255).astype(np.uint8).tolist()
        color = (0,0,255)

        cv2.rectangle(img, (group_x0, group_y0), (group_x1, group_y1), color, 2)

    for i, group in enumerate(forming_groups):
        person = group.pop()
        group_x0 = boxes[person][0]
        group_y0 = boxes[person][1]
        group_x1 = boxes[person][2]
        group_y1 = boxes[person][3]

        for person_id in group:
            box = boxes[person_id]
            if box is None:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            if x0 < group_x0:
                group_x0 = x0
            if y0 < group_y0:
                group_y0 = y0
            if x1 > group_x1:
                group_x1 = x1
            if y1 > group_y1:
                group_y1 = y1

        color_ = _COLORS[i % _COLORS.shape[0]]
        color = (color_ * 255).astype(np.uint8).tolist()
        color = (0,255,0)

        cv2.rectangle(img, (group_x0, group_y0), (group_x1, group_y1), color, 2)


    ids_list = [leaving_person_ids, leave_ids, enter_ids]
    for j, ids in enumerate(ids_list):
        for i, person_id in enumerate(ids):
            if j <= 1:
                box = last_boxes[person_id]
            else:
                box = boxes[person_id]
            if box is None:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            if j == 0:
                color = (255,0,0)
            elif j == 1:
                color = (255,255,0)
            elif j == 2:
                color = (0,255,255)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    cv2.putText(img, nbs_in_groups, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
    cv2.putText(img, nbs_not_in_groups, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
    return img