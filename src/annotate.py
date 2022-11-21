from PIL import ImageDraw, ImageFont


def annotate_img(image, boxes):
    '''Annotate image with rectangles and text.

    Takes the original image and the boxes and draws a rectangle
    around the detected objects. For each rectangle a small text
    is drawn on the top with info about the label and the
    confidence (in percentage).

    Parameters
    ----------
    image: PIL.Image
        The original image.
    boxes: list
        The list with the labels, probability and bounding boxes.

    Returns
    ----------
    image: PIL.Image
        The annotated image.
    '''
    # get image size
    width = image.size[0]
    height = image.size[1]

    # list of labels
    labels = list(set([box[0] for box in boxes]))

    # colors for rectangles
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray']

    # draw
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        
        # get label and confidence
        label = box[0]
        confidence = box[1]

        # dimensions of box
        xmin, xmax, ymin, ymax = box[2:]
        
        # limit vertices to image
        xmin = max(xmin, 0)
        xmax = min(xmax, width - 1)
        ymin = max(ymin, 0)
        ymax = min(ymax, height - 1)

        # get color
        color = colors[labels.index(label)%len(colors)]

        # draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=5)

        # draw text
        txt_size = min(15, int(height/40))
        font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', txt_size)
        ymin = max(ymin, txt_size)
        text = f'{label}-{confidence*100:.1f}%'
        draw.text((xmin, ymin), text, anchor='lb', font=font, fill=color)

    return image