from flask import Flask, render_template, request
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)

def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

num_classes = len(class_labels)
model = build_model(num_classes)
model.load_weights('plant_weights.weights.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_plant(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100
    plant_name = class_labels[str(class_index)]
    return plant_name, confidence

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    plant_name, confidence = predict_plant(filepath)
    return render_template('result.html', plant_name=plant_name, confidence=round(confidence, 2), image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)