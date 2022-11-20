from PIL import ImageDraw, ImageFont


def annotate_img(image, boxes, image_size):

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
        xmax = min(xmax, image_size[0] - 1)
        ymin = max(ymin, 0)
        ymax = min(ymax, image_size[1] - 1)

        # get color
        color = colors[labels.index(label)%len(colors)]

        # draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=5)

        # draw text
        txt_size = min(15, int(image_size[1]/40))
        font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', txt_size)
        ymin = max(ymin, txt_size)
        text = label + '-' + '{:.1f}'.format(confidence*100) + '%'
        draw.text((xmin, ymin), text, anchor='lb', font=font, fill=color)

    return image