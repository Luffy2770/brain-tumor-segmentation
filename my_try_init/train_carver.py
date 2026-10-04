import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set deterministic random seeds for full reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from config import (
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, DEVICE, IMG_SIZE, 
    SEG_BATCH_SIZE, SEG_LR, SEG_EPOCHS, SEG_PATIENCE, 
    CARVER_WEIGHTS, LOSS_TYPE, USE_AUGMENTATION
)
from s5 import AttentionUNet2D
from losses import FocalTverskyLoss, TverskyLoss, DiceLoss

class SegmentationDataset(Dataset):
    """
    Feeds 2D image and mask slices into the Attention U-Net.
    """
    def __init__(self, images, masks, img_size=IMG_SIZE, augment=False):
        self.images = images
        self.masks = masks
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).unsqueeze(0).float()
        mask = torch.from_numpy(np.array(self.masks[idx])).unsqueeze(0).float()

        # Resize to uniform 128x128
        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                            mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                             mode='nearest').squeeze(0)

        # Real-time Data Augmentation to prevent overfitting
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[1])
            if torch.rand(1).item() > 0.5:
                k = torch.randint(1, 4, (1,)).item()
                img = torch.rot90(img, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])

        return img, mask


def calculate_dice(logits, targets, smooth=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    probs, targets = probs.view(-1), targets.view(-1)
    intersection = (probs * targets).sum()
    return ((2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)).item()


def train_attention_unet():
    train_img_path = os.path.join(DATA_DIR, "seg_train_images.npy")
    train_msk_path = os.path.join(DATA_DIR, "seg_train_masks.npy")
    val_img_path = os.path.join(DATA_DIR, "seg_val_images.npy")
    val_msk_path = os.path.join(DATA_DIR, "seg_val_masks.npy")

    if not os.path.exists(train_img_path):
        print("[ERROR] Dataset arrays not found in data/ folder. Running prepare_data.py first...")
        from prepare_data import extract_and_cache_dataset
        extract_and_cache_dataset()

    print(f"============================================================")
    print(f"[TRAINING] Station 5: Attention U-Net from Scratch")
    print(f"   Device         : {DEVICE}")
    print(f"   Loss Function  : {LOSS_TYPE.upper()}")
    print(f"   Batch Size     : {SEG_BATCH_SIZE}")
    print(f"   Learning Rate  : {SEG_LR}")
    print(f"   Max Epochs     : {SEG_EPOCHS}")
    print(f"============================================================")

    # Load arrays with memory mapping for efficiency
    train_images = np.load(train_img_path, mmap_mode='r')
    train_masks  = np.load(train_msk_path, mmap_mode='r')
    val_images   = np.load(val_img_path, mmap_mode='r')
    val_masks    = np.load(val_msk_path, mmap_mode='r')

    print(f"Dataset Loaded: {len(train_images)} Train Slices | {len(val_images)} Val Slices")

    train_ds = SegmentationDataset(train_images, train_masks, IMG_SIZE, augment=USE_AUGMENTATION)
    val_ds   = SegmentationDataset(val_images, val_masks, IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = AttentionUNet2D(in_channels=1, out_channels=1).to(DEVICE)

    # Loss selector
    if LOSS_TYPE == "focal_tversky":
        criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    elif LOSS_TYPE == "tversky":
        criterion = TverskyLoss(alpha=0.3, beta=0.7)
    else:
        criterion = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=SEG_LR)

    best_val_loss = float('inf')
    best_val_dice = 0.0
    patience_counter = 0

    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(SEG_EPOCHS):
        # 1. Training Pass
        model.train()
        running_train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # 2. Validation Pass
        model.eval()
        running_val_loss = 0.0
        running_val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item() * imgs.size(0)
                running_val_dice += calculate_dice(outputs, masks) * imgs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_dice = running_val_dice / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_dices.append(epoch_val_dice)

        print(f"Epoch [{epoch+1:02d}/{SEG_EPOCHS:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Dice: {epoch_val_dice:.4f} ({epoch_val_dice*100:.1f}%)")

        # Save Best Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_dice = epoch_val_dice
            patience_counter = 0
            torch.save(model.state_dict(), CARVER_WEIGHTS)
            print(f"   [+] New Best Model Saved -> {CARVER_WEIGHTS} (Dice: {epoch_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= SEG_PATIENCE:
                print(f"[EARLY STOPPING] Validation loss stopped improving for {SEG_PATIENCE} epochs.")
                break

    # Plot & Save Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(val_losses, label="Val Loss", color="crimson", linewidth=2)
    ax1.set_title(f"{LOSS_TYPE.capitalize()} Loss Progression", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(val_dices, label="Val Dice Score", color="forestgreen", linewidth=2)
    ax2.set_title("Validation Dice Score", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curve_path = os.path.join(OUTPUTS_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"\n[SUCCESS] Training Complete! Best Val Dice: {best_val_dice*100:.2f}%")
    print(f"          Training curves saved -> {curve_path}")

if __name__ == "__main__":
    train_attention_unet()
