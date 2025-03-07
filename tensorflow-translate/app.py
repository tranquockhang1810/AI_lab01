import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS

model_name = "https://tfhub.dev/google/translate_en_vi/1"
translator = tf.safe_model.load(model_name)

app = Flask(__name__)
CORS(app)

@app.route('/translate', methods=['POST'])
def translate():
  data = request.json
  text = data['text']

  if not text:
    return jsonify({"error": "Text is required"}), 400

  result = translator(tf.constant([text])).numpy()[0].decode('utf-8')
  return jsonify({"translated_text": result})


if __name__ == '__main__':  
  app.run(debug=True)