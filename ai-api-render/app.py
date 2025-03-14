from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
  try:
    if 'file' not in request.files:
      return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    result = [{"label": pred[1], "confidence": float(pred[2])} for pred in decoded_predictions]
    return jsonify({'predictions': result})
  
  except Exception as e:
    return jsonify({'error': str(e)})

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=5000)