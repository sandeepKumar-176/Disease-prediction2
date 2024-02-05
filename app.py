from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
import base64

app = Flask(__name__)

# Load the SavedModel
model = tf.keras.models.load_model('./saved_model')

# Define a function to preprocess the image
def preprocess_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image

# Define a function to decode base64 image
def decode_base64_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    return image_bytes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'})

    # Decode the base64 image
    image_base64 = data['image']
    image_bytes = decode_base64_image(image_base64)

    # Preprocess the image
    image = preprocess_image(image_bytes)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Perform inference
    prediction = model.predict(image)
    
    # Format and return the prediction
    return jsonify({'prediction': prediction.tolist()})
@app.route('/', methods=['GET'])
def test():
    return jsonify({'data':"hello"})
if __name__ == '__main__':
    app.run(port=3000)
