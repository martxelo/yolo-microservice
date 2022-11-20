import base64
import io

from PIL import Image
import numpy as np

from src import model
from src.annotate import annotate_img
from src.decode import decode_pred, non_max_supression, scale_boxes

yolo_size = 512


def process_img(file):

    image = Image.open(file).convert('RGB')

    # get original size
    img_size = image.size

    # resize
    new_image = np.array(image.resize((yolo_size, yolo_size)))/255
    new_image = np.expand_dims(new_image, axis=0)

    # predict
    pred = model.predict(new_image)

    # remove first axis
    pred = [np.squeeze(x).astype(float) for x in pred]

    # decode prediction
    boxes = decode_pred(pred, yolo_size)

    # non-max supression
    boxes = non_max_supression(boxes)

    # scale to original size
    boxes = scale_boxes(boxes, yolo_size, img_size)

    # create annotations
    image = annotate_img(image, boxes, img_size)

    return image, boxes
