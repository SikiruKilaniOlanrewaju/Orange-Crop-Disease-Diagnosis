import os
from PIL import Image
import torch
import numpy as np
import streamlit as st
from torchvision import transforms
from src.model import OrangeDiseaseCNN
from src.data_processing import get_image_paths_and_labels

# Config
st.set_page_config(page_title="Orange Disease Diagnosis", layout="centered")
st.title("🍊 Orange Leaf Disease Diagnosis")
st.write("Upload an image of an orange leaf to detect disease.")

# Load model and label mappings
DATASET_DIR = os.path.join('data', 'Disease Dataset')
_, labels = get_image_paths_and_labels(DATASET_DIR)
label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
index_to_label = {v: k for k, v in label_to_index.items()}

num_classes = len(label_to_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OrangeDiseaseCNN(num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join('models', 'orange_disease_cnn.pth'), map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Diagnose"):
        with st.spinner('Analyzing...'):
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = index_to_label[predicted.item()]
            st.success(f"Prediction: **{prediction}**")
