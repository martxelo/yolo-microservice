from itertools import product
from operator import itemgetter

import numpy as np


def get_labels():
    '''Get label list.

    Reads the coco classes file and creates a list with
    all the labels.

    Returns
    ----------
    labels: list
        The list with the coco classes.
    '''

    with open('classes/coco_classes.txt', 'r') as f:
        labels = f.readlines()

    labels = [label[:-1] for label in labels]
    
    return labels


def sigmoid(x):
    '''Applies the sigmoid.

    Takes a numpy array and calculates the sigmoid
    of the values (element-wise).

    Parameters
    ----------
    x: np.ndarray
        The input values.
    
    Returns
    ----------
    sigmoid: np.ndarray
        The sigmoid of the values
    ''' 
    return 1.0/(1.0 + np.exp(-x))


def calc_area(xmin, xmax, ymin, ymax):
    '''Calculate the area of a rectangle.

    Calculates the area of the rectangle with the position
    of the edges. Assumes that xmin<xmax and ymin<ymax. Otherwise
    the result is zero (used for intersection over union).

    Parameters
    ----------
    xmin: float
        The left edge.
    xmax: float
        The right edge.
    ymin: float
        The upper edge.
    ymax: float
        The lower edge.
    
    Returns
    ----------
    area: float
        The area of the rectangle.
    '''
    x = max((xmax - xmin), 0)
    y = max((ymax - ymin), 0)

    return x * y


def calc_overlap(box1, box2):
    '''Calculates the overlap of two boxes.

    The overlap is calculated as the intersection over
    union of the boxes. If both boxes are the same the
    result is 1. If they do not overlap the the result
    is 0.

    Parameters
    ----------
    box1: list
        The first detection box.
    box2: list
        The second detection box.
    
    Returns
    ----------
    iou: float
        The intersection over union of the boxes.
    '''

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
    '''Applies non max supression.

    This function takes all the boxes and removes the ones
    that overlap too much with others. If two boxes overlap
    more than the threshold then the one with lower confidence
    is removed from the list.

    Parameters
    ----------
    boxes: list
        The list with the labels, probability and bounding boxes.
    
    Returns
    ----------
    boxes: list
        The filtered list with the labels, probability and bounding boxes.
    '''

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
    '''Scale the values to the original size.

    Takes the boxes from yolo size and scales them
    to the original image size.

    Parameters
    ----------
    boxes: list
        The list with the labels, probability and bounding boxes.
    size: int
        The size of the yolo input.
    img_size: tuple
        The size of the original image. 
    
    Returns
    ----------
    boxes: list
        The scaled list with the labels, probability and bounding boxes.
    '''
    # get the scale for both dimensions
    scale_x = img_size[0]/size
    scale_y = img_size[1]/size

    # empty list
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


def decode_pred(pred_img, yolo_size, threshold=0.5):
    '''Decode the prediction values.

    YOLO gives three tensors with different number of pixels.
    Each one has three stacked anchors. An anchor is a 3D tensor
    with the information of a detection for each pixel.

    This function takes the tensors and decode the prediction values.
    The position, the dimensions, the confidence and the detected
    class.

    Parameters
    ----------
    pred_img: list
        The list the three predictions.
    yolo_size: int
        The size of the yolo input.
    threshold: float
        The threshold value for consider a true detection.
    
    Returns
    ----------
    boxes: list
        The list with the labels, probability and bounding boxes.    
    '''

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