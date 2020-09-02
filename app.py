import os
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
from keras.models import load_model
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# STATIC_FOLDER = 'static'
# # Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = 'uploads'
# # Path to the folder where we store the different models
MODEL_FOLDER = 'models'
#
#
# def load__model():
#     """Load model once at running time for all the predictions"""
#     print('[INFO] : Model loading ................')
#     global model
#     # model = tf.keras.models.load_model(MODEL_FOLDER + '/catsVSdogs.h5')
#     model = load_model(MODEL_FOLDER + '/cat_dog_classifier.h5')
#     global graph
#     graph = tf.get_default_graph()
#     print('[INFO] : Model loaded')
#
#
def predict(fullpath):
    classifier = load_model('./models/catordog.h5')
    test_image = image.load_img(fullpath, target_size=(160, 160))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    predictions = classifier.predict(test_image).flatten()

    return predictions

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict(fullname)
        print(result)
        pred_prob = result.item()

        if pred_prob > .5:
            label = 'Dog'
            accuracy = round(pred_prob * 10, 2)
        else:
            label = 'Cat'
            accuracy = round((1 - pred_prob) * 10, 2)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')