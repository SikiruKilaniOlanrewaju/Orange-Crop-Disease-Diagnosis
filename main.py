# Entry point for Orange Crop Disease Diagnosis System

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from data_processing import get_image_paths_and_labels, preview_images, split_dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.model import OrangeDiseaseCNN
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

class OrangeDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_index, img_size=(128, 128)):
        self.image_paths = image_paths
        self.labels = [label_to_index[y] for y in labels]
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def main():
    dataset_dir = os.path.join('data', 'Disease Dataset')
    image_paths, labels = get_image_paths_and_labels(dataset_dir)
    print(f"Found {len(image_paths)} images in {len(set(labels))} classes.")
    preview_images(image_paths, labels, n=5)
    X_train_paths, X_test_paths, y_train, y_test = split_dataset(image_paths, labels)
    print(f"Training set: {len(X_train_paths)} images, Test set: {len(X_test_paths)} images")

    label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    num_classes = len(label_to_index)

    train_dataset = OrangeDataset(X_train_paths, y_train, label_to_index)
    test_dataset = OrangeDataset(X_test_paths, y_test, label_to_index)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OrangeDiseaseCNN(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop (1 epoch for demo)
    model.train()
    for epoch in range(1):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join('models', 'orange_disease_cnn.pth'))
    print("Model training complete and saved.")

    # Evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=sorted(label_to_index, key=label_to_index.get)))

    # Visualize confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_to_index))
    plt.xticks(tick_marks, sorted(label_to_index, key=label_to_index.get), rotation=45)
    plt.yticks(tick_marks, sorted(label_to_index, key=label_to_index.get))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Visualize some predictions
    n = 5
    plt.figure(figsize=(15, 3))
    for i, (img_path, pred, true) in enumerate(zip(X_test_paths[:n], all_preds[:n], all_labels[:n])):
        img = Image.open(img_path)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"Pred: {list(label_to_index.keys())[list(label_to_index.values()).index(pred)]}\nTrue: {list(label_to_index.keys())[list(label_to_index.values()).index(true)]}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
