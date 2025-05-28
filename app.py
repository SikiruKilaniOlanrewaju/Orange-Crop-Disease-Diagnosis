from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from src.model import OrangeDiseaseCNN
from src.data_processing import get_image_paths_and_labels

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load label mapping
DATASET_DIR = os.path.join('data', 'Disease Dataset')
_, labels = get_image_paths_and_labels(DATASET_DIR)
label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
index_to_label = {v: k for k, v in label_to_index.items()}

# Load model
num_classes = len(label_to_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OrangeDiseaseCNN(num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join('models', 'orange_disease_cnn.pth'), map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Predict
            img = Image.open(filepath).convert('RGB')
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs.data, 1)
                prediction = index_to_label[predicted.item()]
            os.remove(filepath)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
