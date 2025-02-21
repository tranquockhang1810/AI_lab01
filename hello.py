import tensorflow_hub as hub
import tensorflow as tf
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt

model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4" 
model = hub.load(model_url)
print("Model loaded successfully!")

image_path = "https://www.thesprucepets.com/thmb/RvdTRaFSbKoIbbCfkLwOPr2H8pk=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/GettyImages-991139016-493253fd4ce344b4bd160b7c3a6703a5.jpg"  # URL hình ảnh 
image = Image.open(tf.keras.utils.get_file("b.jpg", image_path)).resize((224, 224))

def preprocess_image(image): 
    image = np.array(image).astype(np.float32) / 255.0   # Chuẩn hóa pixel về khoảng [0, 1] 
    return image[np.newaxis, ...]  # Thêm batch dimension 
processed_image = preprocess_image(image) 
print("Image preprocessed successfully!")

predictions = model(processed_image) 
predicted_class = np.argmax(predictions, axis=-1) 
print("Predicted class index:", predicted_class) 
# Hàm np.argmax tìm chỉ số của lớp có xác suất cao nhất. 
# Tải danh sách nhãn lớp từ ImageNet: 
labels_path = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt" 
labels = tf.keras.utils.get_file("ImageNetLabels.txt", labels_path) 
with open(labels, "r") as f: 
    labels = f.read().splitlines() 
print("Predicted label:", labels[predicted_class[0]]) 

plt.imshow(image) 
plt.title(f"Prediction: {labels[predicted_class[0]]}") 
plt.axis('off') 
plt.show()