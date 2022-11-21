from PIL import Image
import numpy as np

from src import model
from src.annotate import annotate_img
from src.decode import decode_pred, non_max_supression, scale_boxes

yolo_size = 512


def process_img(file):
    '''Process and image.

    Takes a image file, resizes it to a fixed size and feeds it to
    the YOLOV3 neural network. Then gets the prediction and process
    it:
    - Decode the tensor output information.
    - Applies non max supression.
    - Scale the result to the original size.
    
    Parameters
    ----------
    file: BufferedReader
        The with the image information.
    
    Returns
    ----------
    image: PIL.Image
        The annotated image.
    boxes: list
        The list with the labels, probability and bounding boxes.
    '''
    # read file
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
    image = annotate_img(image, boxes)

    return image, boxes
