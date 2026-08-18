import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    PROCESSED_DIR, DEVICE, IMG_SIZE, SEG_BATCH_SIZE, 
    SEG_LR, SEG_EPOCHS, SEG_PATIENCE, UNET_MODEL_PATH,
    ATTENTION_UNET_MODEL_PATH, MODEL_TYPE, LOSS_TYPE, USE_DATA_AUGMENTATION
)
from unet_model import UNet2D
from attention_unet import AttentionUNet2D
from dice_loss import DiceLoss
from losses import FocalTverskyLoss, TverskyLoss, ComboLoss


class SegDataset(Dataset):
    def __init__(self, images, masks, img_size=IMG_SIZE, augment=False):
        self.images = images
        self.masks = masks
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(np.array(self.images[i])).unsqueeze(0).float()
        mask = torch.from_numpy(np.array(self.masks[i])).unsqueeze(0).float()

        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                             mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode='nearest').squeeze(0)

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


def dice_score(logits, targets, smooth=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    probs, targets = probs.view(-1), targets.view(-1)
    intersection = (probs * targets).sum()
    return ((2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)).item()


if __name__ == "__main__":
    train_file = f"{PROCESSED_DIR}/seg_train_images.npy"
    if not os.path.exists(train_file):
        print("Training array seg_train_images.npy not found.")
        sys.exit(0)

    train_images = np.load(train_file, mmap_mode='r')
    train_masks  = np.load(f"{PROCESSED_DIR}/seg_train_masks.npy", mmap_mode='r')
    val_images   = np.load(f"{PROCESSED_DIR}/seg_val_images.npy", mmap_mode='r')
    val_masks    = np.load(f"{PROCESSED_DIR}/seg_val_masks.npy", mmap_mode='r')

    print(f"Train: {len(train_images)} slices | Val: {len(val_images)} slices", flush=True)
    print(f"Model Type: {MODEL_TYPE.upper()} | Loss Type: {LOSS_TYPE.upper()} | Augmentations: {USE_DATA_AUGMENTATION}", flush=True)

    train_ds = SegDataset(train_images, train_masks, IMG_SIZE, augment=USE_DATA_AUGMENTATION)
    val_ds   = SegDataset(val_images, val_masks, IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=0)

    if MODEL_TYPE == "attention_unet":
        model = AttentionUNet2D(in_channels=1, out_channels=1).to(DEVICE)
        save_checkpoint = ATTENTION_UNET_MODEL_PATH
    else:
        model = UNet2D(in_channels=1, out_channels=1).to(DEVICE)
        save_checkpoint = UNET_MODEL_PATH

    if LOSS_TYPE == "focal_tversky":
        criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    elif LOSS_TYPE == "tversky":
        criterion = TverskyLoss(alpha=0.3, beta=0.7)
    elif LOSS_TYPE == "combo":
        criterion = ComboLoss()
    else:
        criterion = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=SEG_LR)

    EPOCHS = SEG_EPOCHS
    train_losses, val_losses, val_dices = [], [], []
    best_val_loss = float('inf')
    patience, patience_counter = SEG_PATIENCE, 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss, val_running_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item() * imgs.size(0)
                val_running_dice += dice_score(outputs, masks) * imgs.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_dice = val_running_dice / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        print(f"Epoch {epoch+1}/{EPOCHS} -- train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_dice: {val_dice:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_checkpoint)
            print(f"   [+] Saved best model -> {save_checkpoint}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.", flush=True)
                break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title(f"{LOSS_TYPE.capitalize()} Loss"); ax1.legend()
    ax2.plot(val_dices, color='green', label="Val Dice Score")
    ax2.set_title("Validation Dice Score"); ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/seg_training_curves.png")
    plt.close(fig)