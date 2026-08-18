import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROCESSED_DIR, DEVICE, IMG_SIZE, SEG_BATCH_SIZE
from utils import load_segmenter_model

def dice_score(logits, targets, smooth=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    probs, targets = probs.view(-1), targets.view(-1)
    intersection = (probs * targets).sum()
    return ((2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)).item()


def run_evaluation_seg():
    test_img_file = f"{PROCESSED_DIR}/seg_test_images.npy"
    if not os.path.exists(test_img_file):
        print("Evaluation file seg_test_images.npy not found.")
        return

    test_images = np.load(test_img_file, mmap_mode='r')
    test_masks  = np.load(f"{PROCESSED_DIR}/seg_test_masks.npy", mmap_mode='r')

    from train_seg import SegDataset
    test_ds = SegDataset(test_images, test_masks, IMG_SIZE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=0)

    model = load_segmenter_model(DEVICE)

    all_dice = []
    with torch.inference_mode():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            for i in range(imgs.size(0)):
                d = dice_score(outputs[i:i+1], masks[i:i+1])
                all_dice.append(d)

    all_dice = np.array(all_dice)
    print(f"\n--- Test Set Results (n={len(all_dice)} tumor-slices) ---")
    print(f"Mean Dice:   {all_dice.mean():.4f}")
    print(f"Median Dice: {np.median(all_dice):.4f}")
    print(f"Min Dice:    {all_dice.min():.4f}")
    print(f"Max Dice:    {all_dice.max():.4f}")
    print(f"Std Dice:    {all_dice.std():.4f}")

    fig1 = plt.figure(figsize=(8, 5))
    plt.hist(all_dice, bins=30, color="steelblue", edgecolor="white")
    plt.axvline(all_dice.mean(), color="coral", linestyle="--", label=f"mean={all_dice.mean():.3f}")
    plt.xlabel("Dice Score"); plt.ylabel("Number of slices")
    plt.title("Test Set Dice Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/seg_test_dice_distribution.png")
    plt.close(fig1)

    sample_indices = np.random.choice(len(test_images), 4, replace=False)
    fig2, axes = plt.subplots(4, 3, figsize=(9, 12))
    with torch.inference_mode():
        for row, idx in enumerate(sample_indices):
            img = torch.from_numpy(np.array(test_images[idx]))[None, None, ...].float().to(DEVICE)
            img_r = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            pred = torch.sigmoid(model(img_r))[0, 0].cpu().numpy()
            true_mask = np.array(test_masks[idx])

            axes[row, 0].imshow(test_images[idx], cmap="gray")
            axes[row, 0].set_title("Input Slice"); axes[row, 0].axis("off")
            axes[row, 1].imshow(true_mask, cmap="Reds")
            axes[row, 1].set_title("Ground Truth"); axes[row, 1].axis("off")
            axes[row, 2].imshow(pred > 0.5, cmap="Reds")
            axes[row, 2].set_title("Prediction"); axes[row, 2].axis("off")
    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/seg_prediction_samples.png")
    plt.close(fig2)


if __name__ == "__main__":
    run_evaluation_seg()