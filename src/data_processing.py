# data_processing.py
# Functions to load, preview, and split image dataset for Orange Crop Disease Diagnosis System

import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

def get_image_paths_and_labels(dataset_dir):
    """
    Scans the dataset directory and returns lists of image file paths and their corresponding labels.
    """
    image_paths = []
    labels = []
    for label_name in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_name)
        if os.path.isdir(label_path):
            for fname in os.listdir(label_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(label_path, fname))
                    labels.append(label_name)
    return image_paths, labels

def preview_images(image_paths, labels, n=5):
    """
    Displays n sample images with their labels.
    """
    plt.figure(figsize=(15, 3))
    for i in range(min(n, len(image_paths))):
        img = Image.open(image_paths[i])
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

def split_dataset(image_paths, labels, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    return train_test_split(image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels)
