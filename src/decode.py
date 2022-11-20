from itertools import product
from operator import itemgetter

import numpy as np


def get_labels():

    with open('classes/coco_classes.txt', 'r') as f:
        labels = f.readlines()

    labels = [label[:-1] for label in labels]
    
    return labels


def sigmoid(x):
    
    return 1.0/(1.0 + np.exp(-x))


def calc_area(x1, x2, y1, y2):

    x = max((x2 - x1), 0)
    y = max((y2 - y1), 0)

    return x * y


def calc_overlap(box1, box2):

    # box 1
    xmin1 = box1[2]
    xmax1 = box1[3]
    ymin1 = box1[4]
    ymax1 = box1[5]

    # box 1
    xmin2 = box2[2]
    xmax2 = box2[3]
    ymin2 = box2[4]
    ymax2 = box2[5]

    # intersection coordinates
    ixmin = max(xmin1, xmin2)
    ixmax = min(xmax1, xmax2)
    iymin = max(ymin1, ymin2)
    iymax = min(ymax1, ymax2)

    # areas
    area_1 = calc_area(xmin1, xmax1, ymin1, ymax1)
    area_2 = calc_area(xmin2, xmax2, ymin2, ymax2)
    area_in = calc_area(ixmin, ixmax, iymin, iymax)

    return area_in/(area_1 + area_2 - area_in)


def non_max_supression(boxes, max_overlap=0.25):

    # sort by confidence
    boxes = sorted(boxes, key=itemgetter(1,3), reverse=True)

    # empty list
    final_boxes = []
    while len(boxes)>0:

        # add the box with highest confidence
        max_conf_box = boxes.pop(0)
        final_boxes.append(max_conf_box)

        remove_idx = []
        for i, box in enumerate(boxes):

            overlap = calc_overlap(max_conf_box, box)

            if overlap > max_overlap:
                remove_idx.append(i)

        # remove 
        boxes = [boxes[i] for i in range(len(boxes)) if i not in remove_idx]

    return final_boxes


def scale_boxes(boxes, size, img_size):

    scale_x = img_size[0]/size
    scale_y = img_size[1]/size

    scaled_boxes = []

    for box in boxes:

        label = box[0]
        confidence = box[1]
        xmin = box[2]*scale_x
        xmax = box[3]*scale_x
        ymin = box[4]*scale_y
        ymax = box[5]*scale_y

        scaled_boxes.append([label, confidence, xmin, xmax, ymin, ymax])

    return scaled_boxes


def decode_pred(pred_img, yolo_size, threshold = 0.5):

    labels = get_labels()

    anchors = [
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [59, 119]],
        [[10, 13], [16, 30], [33, 23]]
        ]

    n_anchors = len(anchors)
    n_preds = len(pred_img)
    boxes = []

    for n_pred, n_anch in product(range(n_preds), range(n_anchors)):
        
        pred = pred_img[n_pred]

        n_cells_x = pred.shape[0]
        n_cells_y = pred.shape[1]
        cell_size_x = int(yolo_size/n_cells_x)
        cell_size_y = int(yolo_size/n_cells_y)

        anchor = pred[:, :, n_anch*85:(n_anch+1)*85]
        detection = sigmoid(anchor[:, :, 4])

        for cell_x, cell_y in product(range(n_cells_x), range(n_cells_y)):

            confidence = detection[cell_x, cell_y]

            if confidence > threshold:
                # calculate center
                x = cell_size_x*(cell_x + sigmoid(anchor[cell_x, cell_y, 0]))
                y = cell_size_y*(cell_y + sigmoid(anchor[cell_x, cell_y, 1]))

                # calculate width and height
                scale_w = anchors[n_pred][n_anch][0]
                scale_h = anchors[n_pred][n_anch][1]
                w = scale_w*np.exp(anchor[cell_x, cell_y, 2])
                h = scale_h*np.exp(anchor[cell_x, cell_y, 3])

                # get label
                label = labels[np.argmax(anchor[cell_x, cell_y, 5:])]

                # convert to vertices
                xmin = y - w/2
                xmax = y + w/2
                ymin = x - h/2
                ymax = x + h/2

                # add to list of boxes
                boxes.append([label, confidence, xmin, xmax, ymin, ymax])
    
    return boxes