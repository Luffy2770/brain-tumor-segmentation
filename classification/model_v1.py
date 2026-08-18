import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    PROCESSED_DIR, DEVICE, IMG_SIZE, CLASSIFIER_BATCH_SIZE, 
    CLASSIFIER_LR, CLASSIFIER_EPOCHS, CLASSIFIER_PATIENCE, CLASSIFIER_MODEL_PATH
)
from classification.model import TumorClassifierCNN

DATA_DIR = PROCESSED_DIR
device = DEVICE

class MRISliceDataset(Dataset):
    def __init__(self, images, labels, indices, img_size=128):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = np.array(self.images[idx])
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                             mode='bilinear', align_corners=False).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def train_classifier():
    img_file = f"{DATA_DIR}/images.npy"
    if not os.path.exists(img_file):
        print("Dataset array images.npy not found for training.")
        return

    images = np.load(img_file, mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/labels.npy")
    patient_ids = np.load(f"{DATA_DIR}/patient_ids.npy", allow_pickle=True)

    unique_patients = np.unique(patient_ids)
    train_p, temp_p = train_test_split(unique_patients, test_size=0.2, random_state=42)
    val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

    def get_indices(patient_subset):
        return np.where(np.isin(patient_ids, patient_subset))[0]

    train_idx = get_indices(train_p)
    val_idx = get_indices(val_p)
    test_idx = get_indices(test_p)

    train_ds = MRISliceDataset(images, labels, train_idx, img_size=IMG_SIZE)
    val_ds   = MRISliceDataset(images, labels, val_idx, img_size=IMG_SIZE)
    test_ds  = MRISliceDataset(images, labels, test_idx, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=0)

    model = TumorClassifierCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = CLASSIFIER_EPOCHS
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = CLASSIFIER_PATIENCE, 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_running_loss += loss.item() * imgs.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} -- train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CLASSIFIER_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(f"{DATA_DIR}/loss_curve.png")
    plt.close()

    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(lbls.numpy())

    print("\n--- Test Set Results ---")
    print(classification_report(all_true, all_preds, target_names=["No Tumor", "Tumor"]))
    print(confusion_matrix(all_true, all_preds))


if __name__ == "__main__":
    train_classifier()