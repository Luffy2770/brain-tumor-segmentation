import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PROCESSED_DIR, DEVICE, CHOSEN_THRESHOLD, CLASSIFIER_MODEL_PATH, IMG_SIZE
from classification.model import TumorClassifierCNN

DATA_DIR = PROCESSED_DIR
device = DEVICE

class MRISliceDataset(Dataset):
    def __init__(self, images, labels, indices, img_size=IMG_SIZE):
        self.images, self.labels, self.indices, self.img_size = images, labels, indices, img_size
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        img = torch.from_numpy(np.array(self.images[idx])).unsqueeze(0).float()
        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                             mode='bilinear', align_corners=False).squeeze(0)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def run_analysis_v1():
    img_file = f"{DATA_DIR}/images.npy"
    if not os.path.exists(img_file):
        print("Data file images.npy not found for offline evaluation.")
        return

    images = np.load(img_file, mmap_mode='r')
    labels = np.load(f"{DATA_DIR}/labels.npy")
    patient_ids = np.load(f"{DATA_DIR}/patient_ids.npy", allow_pickle=True)

    unique_patients = np.unique(patient_ids)
    train_p, temp_p = train_test_split(unique_patients, test_size=0.2, random_state=42)
    val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
    test_idx = np.where(np.isin(patient_ids, test_p))[0]

    test_ds = MRISliceDataset(images, labels, test_idx)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = TumorClassifierCNN().to(device)
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    model.eval()

    all_probs, all_true = [], []
    with torch.inference_mode():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(lbls.numpy())
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    preds_default = (all_probs > 0.5).astype(int)
    cm = confusion_matrix(all_true, preds_default)
    fig1 = plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Tumor","Tumor"], yticklabels=["No Tumor","Tumor"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/confusion_matrix.png")
    plt.close(fig1)

    fpr, tpr, thresholds = roc_curve(all_true, all_probs)
    roc_auc = auc(fpr, tpr)
    fig2 = plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/roc_curve.png")
    plt.close(fig2)

    thresh_range = np.arange(0.1, 0.9, 0.05)
    precisions, recalls = [], []
    for t in thresh_range:
        preds_t = (all_probs > t).astype(int)
        tp = ((preds_t==1)&(all_true==1)).sum()
        fp = ((preds_t==1)&(all_true==0)).sum()
        fn = ((preds_t==0)&(all_true==1)).sum()
        precisions.append(tp/(tp+fp+1e-8))
        recalls.append(tp/(tp+fn+1e-8))

    fig3 = plt.figure(figsize=(7,5))
    plt.plot(thresh_range, precisions, label="Precision (Tumor)")
    plt.plot(thresh_range, recalls, label="Recall (Tumor)")
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label="default 0.5")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.legend()
    plt.title("Precision/Recall vs Decision Threshold")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/threshold_tradeoff.png")
    plt.close(fig3)

    preds_final = (all_probs > CHOSEN_THRESHOLD).astype(int)
    print(f"\n--- Results at threshold = {CHOSEN_THRESHOLD} ---")
    print(classification_report(all_true, preds_final, target_names=["No Tumor","Tumor"]))


if __name__ == "__main__":
    run_analysis_v1()