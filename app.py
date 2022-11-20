import base64

from flask import Flask, request, jsonify

from yolo.process import process_img
from yolo.model import start_model

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():

    # read image
    image = request.files['image']

    # process the image
    image, boxes = process_img(image)

    # make it jsonizable
    size = image.size
    data = base64.b64encode(image.tobytes()).decode()
    
    return jsonify({'data': data, 'size':size, 'boxes': boxes})


if __name__ == '__main__':

    start_model()

    app.run(port=3674)
