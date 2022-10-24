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
        x, y, w, h = box[2:]
        
        # calculate vertices
        x0 = max(y - w/2, 0)
        x1 = min(y + w/2, image_size[0] - 1)
        y0 = max(x - h/2, 0)
        y1 = min(x + h/2, image_size[1] - 1)

        # get color
        color = colors[labels.index(label)%len(colors)]

        # draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color)

        # draw text
        txt_size = min(10, int(image_size[1]/40))
        font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', txt_size)
        y0 = max(y0, txt_size)
        text = label + '-' + '{:.1f}'.format(confidence*100) + '%'
        draw.text((x0, y0), text, anchor='lb', font=font, fill=color)

    return image