from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

app = Flask(__name__)
CORS(app)

HUGGING_FACE_ANALYZE_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HUGGING_FACE_TRANSLATE_API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-vi"
HEADERS = {"Authorization": "Bearer " + os.getenv("HUGGING_FACE_API_KEY")}

@app.route('/analyze', methods=['POST'])
def analyze():
  data = request.json
  text = data['text']
  response = requests.post(HUGGING_FACE_ANALYZE_API_URL, headers=HEADERS, json={"inputs": text})
  return jsonify(response.json())

@app.route('/translate', methods=['POST'])
def translate():
  data = request.json
  text = data['text']

  if not text:
    return jsonify({"error": "Text is required"}), 400
  
  response = requests.post(HUGGING_FACE_TRANSLATE_API_URL, headers=HEADERS, json={"inputs": text})
  return jsonify(response.json())

if __name__ == '__main__':
  app.run(debug=True)
