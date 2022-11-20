from tensorflow.keras.models import load_model

MODEL = 0

def predict(x):

    return MODEL.predict(x)
    

def start_model():

    global MODEL
    
    MODEL = load_model('weights/yolov3.h5')