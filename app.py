from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Cargar el modelo
model = load_model('my_model.h5')

def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)
    predictions = model.predict(image)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
