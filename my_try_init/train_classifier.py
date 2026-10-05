import os
import random
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Enforce deterministic random seeds for full reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from config import (
    DATASET_DIR, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, DEVICE, IMG_SIZE,
    CLASSIFIER_BATCH_SIZE, CLASSIFIER_LR, CLASSIFIER_EPOCHS,
    CLASSIFIER_PATIENCE, CLASSIFIER_WEIGHTS
)
from s1 import ScanLoader
from s2 import SliceNormalizer
from s4 import TumorClassifierCNN


def extract_patient_slices(patient_list, split_name, img_size=IMG_SIZE):
    """
    Extracts all non-empty 2D slices (both tumor and healthy brain) from raw NIfTI files
    for a strictly defined list of patient IDs. Resizes directly to img_size (128x128).
    Uses the patched SliceNormalizer (filters slices with near-zero variance).
    """
    normalizer = SliceNormalizer()
    images, labels, pids = [], [], []

    print(f"[{split_name.upper()}] Extracting non-empty slices for {len(patient_list)} patients...")
    for idx, pid in enumerate(patient_list, start=1):
        flair_path = os.path.join(DATASET_DIR, pid, f"{pid}_flair.nii")
        seg_path = os.path.join(DATASET_DIR, pid, f"{pid}_seg.nii")

        if not (os.path.exists(flair_path) and os.path.exists(seg_path)):
            continue

        loader = ScanLoader(flair_path)
        seg_data = nib.load(seg_path).get_fdata()

        for i in range(loader.total_slices):
            raw_slice = loader.get_slice(i)
            if loader.is_empty_slice(raw_slice):
                continue

            norm_slice = normalizer.normalize(raw_slice, mask_background=False)
            if norm_slice is None:
                continue

            has_tumor = int((seg_data[:, :, i] > 0).any())

            # Resize to target 128x128 matching Station 3 Funnel
            t_slice = torch.from_numpy(norm_slice)[None, None, ...].float()
            r_slice = F.interpolate(t_slice, size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze().numpy()

            images.append(r_slice)
            labels.append(has_tumor)
            pids.append(pid)

        if idx % 50 == 0 or idx == len(patient_list):
            print(f"  [{split_name.upper()}] {idx:03d}/{len(patient_list):03d} patients processed ({len(images):,} slices)")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(pids)


def get_classifier_dataset(force_reextract=False):
    """
    Loads or creates patient-partitioned dataset for Stage 4 Classifier.
    Guarantees strict zero-overlap with seg_test_pids.npy.
    """
    cls_train_img_path = os.path.join(DATA_DIR, "cls_train_images.npy")
    cls_train_lbl_path = os.path.join(DATA_DIR, "cls_train_labels.npy")
    cls_train_pid_path = os.path.join(DATA_DIR, "cls_train_pids.npy")

    cls_val_img_path = os.path.join(DATA_DIR, "cls_val_images.npy")
    cls_val_lbl_path = os.path.join(DATA_DIR, "cls_val_labels.npy")
    cls_val_pid_path = os.path.join(DATA_DIR, "cls_val_pids.npy")

    test_pid_path = os.path.join(DATA_DIR, "seg_test_pids.npy")
    if not os.path.exists(test_pid_path):
        raise FileNotFoundError(f"Missing {test_pid_path}. Run prepare_data.py first!")

    test_pids_set = set(np.unique(np.load(test_pid_path)))

    all_exist = all(os.path.exists(p) for p in [
        cls_train_img_path, cls_train_lbl_path, cls_train_pid_path,
        cls_val_img_path, cls_val_lbl_path, cls_val_pid_path
    ])

    if force_reextract or not all_exist:
        print("[DATA] Building clean patient-partitioned slice database with patched SliceNormalizer...")
        seg_train_pids = sorted(list(set(np.load(os.path.join(DATA_DIR, "seg_train_pids.npy")))))
        seg_val_pids   = sorted(list(set(np.load(os.path.join(DATA_DIR, "seg_val_pids.npy")))))

        # Strict patient-level separation audit
        assert len(set(seg_train_pids) & test_pids_set) == 0, "FATAL: Train patients overlap with Test set!"
        assert len(set(seg_val_pids) & test_pids_set) == 0, "FATAL: Val patients overlap with Test set!"

        train_imgs, train_lbls, train_pids = extract_patient_slices(seg_train_pids, "train")
        val_imgs, val_lbls, val_pids       = extract_patient_slices(seg_val_pids, "val")

        np.save(cls_train_img_path, train_imgs)
        np.save(cls_train_lbl_path, train_lbls)
        np.save(cls_train_pid_path, train_pids)

        np.save(cls_val_img_path, val_imgs)
        np.save(cls_val_lbl_path, val_lbls)
        np.save(cls_val_pid_path, val_pids)
        print("[DATA] Saved clean patient-partitioned arrays to disk successfully.")
    else:
        print("[DATA] Loading cached patient-partitioned slice database from disk...")
        train_imgs = np.load(cls_train_img_path, mmap_mode='r')
        train_lbls = np.load(cls_train_lbl_path)
        train_pids = np.load(cls_train_pid_path)

        val_imgs = np.load(cls_val_img_path, mmap_mode='r')
        val_lbls = np.load(cls_val_lbl_path)
        val_pids = np.load(cls_val_pid_path)

    # Report array statistics
    print(f"\n[DATA STATS] Train Array: shape={train_imgs.shape}, min={float(train_imgs.min()):.4f}, max={float(train_imgs.max()):.4f}, abs_max={float(np.max(np.abs(train_imgs))):.4f}")
    print(f"[DATA STATS] Val Array  : shape={val_imgs.shape}, min={float(val_imgs.min()):.4f}, max={float(val_imgs.max()):.4f}, abs_max={float(np.max(np.abs(val_imgs))):.4f}")

    # Formal Leakage Audit
    train_u = set(np.unique(train_pids))
    val_u   = set(np.unique(val_pids))

    print(f"============================================================")
    print(f"[DATA AUDIT] Stage 4 Classifier Split Verification")
    print(f"============================================================")
    print(f" Train Patients  : {len(train_u)} ({len(train_imgs):,} total non-empty slices)")
    print(f" Val Patients    : {len(val_u)} ({len(val_imgs):,} total non-empty slices)")
    print(f" Test Patients   : {len(test_pids_set)} (UNTOUCHED)")
    print(f" Overlap (Train x Test) : {len(train_u & test_pids_set)} (STRICT ZERO)")
    print(f" Overlap (Val x Test)   : {len(val_u & test_pids_set)} (STRICT ZERO)")
    print(f"============================================================\n")

    return (train_imgs, train_lbls), (val_imgs, val_lbls)


class ClassifierDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def train_classifier(force_reextract=False):
    print(f"============================================================")
    print(f"[TRAINING] Station 4: Tumor Classifier CNN from Scratch")
    print(f"   Device         : {DEVICE}")
    print(f"   Batch Size     : {CLASSIFIER_BATCH_SIZE}")
    print(f"   Learning Rate  : {CLASSIFIER_LR}")
    print(f"   Max Epochs     : {CLASSIFIER_EPOCHS}")
    print(f"============================================================")

    (train_imgs, train_lbls), (val_imgs, val_lbls) = get_classifier_dataset(force_reextract=force_reextract)

    train_ds = ClassifierDataset(train_imgs, train_lbls)
    val_ds   = ClassifierDataset(val_imgs, val_lbls)

    train_loader = DataLoader(train_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=0)

    model = TumorClassifierCNN().to(DEVICE)

    # Class weights to handle imbalanced healthy (0) vs tumor (1) slices
    num_neg = int(np.sum(train_lbls == 0))
    num_pos = int(np.sum(train_lbls == 1))
    total_samples = len(train_lbls)
    weight_neg = total_samples / (2.0 * max(num_neg, 1))
    weight_pos = total_samples / (2.0 * max(num_pos, 1))
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(DEVICE)

    print(f"[CLASS BALANCE] Train Non-empty Slices:")
    print(f"  - Healthy (0) : {num_neg:,} slices (weight: {weight_neg:.2f})")
    print(f"  - Tumor   (1) : {num_pos:,} slices (weight: {weight_pos:.2f})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)

    best_val_loss = float('inf')
    patience_counter = 0
    saved_epoch = -1
    history = []

    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        running_train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * imgs.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                running_val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc  = (correct / total) * 100

        is_saved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            saved_epoch = epoch + 1
            is_saved = True
            torch.save(model.state_dict(), CLASSIFIER_WEIGHTS)
            save_msg = f"   [+] Saved Best Classifier -> {CLASSIFIER_WEIGHTS} (Epoch {saved_epoch})"
        else:
            patience_counter += 1
            save_msg = ""

        print(f"Epoch [{epoch+1:02d}/{CLASSIFIER_EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_acc:.2f}%"
              f"{save_msg}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_accuracy": round(float(val_acc), 4),
            "saved_checkpoint": is_saved
        })

        if patience_counter >= CLASSIFIER_PATIENCE:
            print(f"[EARLY STOPPING] Validation loss did not improve for {CLASSIFIER_PATIENCE} epochs.")
            break

    # Save training log to CSV
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df_hist = pd.DataFrame(history)
    csv_log_path = os.path.join(OUTPUTS_DIR, "classifier_training_log.csv")
    df_hist.to_csv(csv_log_path, index=False)
    print(f"\n[+] Saved classifier training log -> {csv_log_path}")

    # Plot clean training curve
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs_range = df_hist["epoch"]
    ax_loss.plot(epochs_range, df_hist["train_loss"], marker='o', label="Train Loss", color="#1f77b4", linewidth=2)
    ax_loss.plot(epochs_range, df_hist["val_loss"], marker='s', label="Val Loss", color="#d62728", linewidth=2)
    if saved_epoch != -1:
        ax_loss.axvline(saved_epoch, linestyle="--", color="forestgreen", linewidth=1.8, label=f"Saved Epoch {saved_epoch} (Loss: {best_val_loss:.4f})")
    ax_loss.set_title("Stage 4 Classifier Loss per Epoch", fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontweight="bold")
    ax_loss.set_ylabel("Cross-Entropy Loss", fontweight="bold")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs_range, df_hist["val_accuracy"], marker='o', label="Val Accuracy (%)", color="#2ca02c", linewidth=2)
    if saved_epoch != -1:
        saved_acc = df_hist.loc[df_hist["epoch"] == saved_epoch, "val_accuracy"].values[0]
        ax_acc.axvline(saved_epoch, linestyle="--", color="forestgreen", linewidth=1.8, label=f"Saved Epoch {saved_epoch} ({saved_acc:.2f}%)")
    ax_acc.set_title("Stage 4 Classifier Validation Accuracy", fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)", fontweight="bold")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUTS_DIR, "classifier_training_curves.png")
    plt.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"[+] Saved classifier training curves -> {plot_path}")

    print(f"\n[SUCCESS] Classifier training complete. Checkpoint saved at: {CLASSIFIER_WEIGHTS} (Epoch {saved_epoch})")


if __name__ == "__main__":
    train_classifier(force_reextract=True)
