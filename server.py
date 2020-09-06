#model
from tensorflow.python.keras.models import load_model
import numpy as np
import os
from PIL import Image


from flask import Flask, request, render_template, make_response



app = Flask(__name__,static_folder='C:\\Users\\Vishwa\\Desktop\\Garbage classifier\\Static')

#model
model = load_model('keras_model.h5')
model._make_predict_function()
def model_predict(img_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    imag = Image.open(img_path)
    imag = imag.resize((224, 224))
    image_array = np.asarray(imag)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    pred = model.predict(data)
    return pred
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        image_file = request.files['imagefile']
        filename = image_file.filename
        filepath = os.path.join('C:\\Users\\Vishwa\\Desktop\\Garbage classifier\\uploads', filename)
        image_file.save(filepath)
        # Make prediction
        preds = model_predict(filepath)
        if preds[0][0]>0.5:
            prediction = "BIO"
        else:
            prediction = "NON-BIO"
    return make_response(prediction,200)


@app.route('/',methods=['GET'])
def index():
    return render_template('myproject.html')

@app.route('/demo',methods=['GET'])
def demo():
    return "working"


app.run()