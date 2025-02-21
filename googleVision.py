from google.cloud import vision
import io
from werkzeug.utils import secure_filename
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
  try:
    if 'file' not in request.files:
      return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
      return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    client = vision.ImageAnnotatorClient()
    with io.open(filepath, 'rb') as image_file:
      content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    label_texts = [label.description for label in labels]

    return jsonify({'labels': label_texts})

  except Exception as e:
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
  app.run(debug=True)