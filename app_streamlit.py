import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from src.model import OrangeDiseaseCNN
from src.data_processing import get_image_paths_and_labels

st.set_page_config(page_title="Orange Crop Disease Diagnosis", layout="centered")

# Custom CSS for a modern, professional look
st.markdown('''
    <style>
    body {
        background-color: #f4f6fb;
    }
    .main {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
        padding: 2rem 2.5rem 2rem 2.5rem;
        margin-top: 2rem;
    }
    .stButton>button {
        background-color: #ff9800;
        color: white;
        border-radius: 8px;
        font-size: 1.1rem;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #fb8c00;
    }
    .stFileUploader>div>div {
        border-radius: 8px;
        border: 2px solid #ff9800;
        background: #fff3e0;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #ff9800; margin-bottom: 0;'>🍊 Orange Crop Disease Diagnosis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-top: 0;'>Upload an image of an orange crop leaf to diagnose the disease.</p>", unsafe_allow_html=True)

# Load label mapping
dataset_dir = os.path.join('data', 'Disease Dataset')
_, labels = get_image_paths_and_labels(dataset_dir)
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

with st.container():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            prediction = index_to_label[predicted.item()]
        st.success(f"Prediction: {prediction}")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; color:#888;'>Model powered by PyTorch & Streamlit</div>", unsafe_allow_html=True)
