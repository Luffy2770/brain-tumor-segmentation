import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classification"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "segmentation"))

from config import (
    BRATS_DIR, OUTPUT_DIR, DEVICE, IMG_SIZE, CHOSEN_THRESHOLD, 
    MIN_FRAGMENT_SIZE, DEFAULT_TEST_PATIENTS, CLASSIFIER_MODEL_PATH
)
from model import TumorClassifierCNN
from utils import normalize_slice, clean_small_fragments, to_tensor_batch, load_segmenter_model

classifier = TumorClassifierCNN().to(DEVICE)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
classifier.eval()

segmenter = load_segmenter_model(DEVICE)
segmenter.eval()


def predict_full_volume(scan, min_fragment_size=MIN_FRAGMENT_SIZE):
    depth = scan.shape[2]
    pred_stack = np.zeros_like(scan, dtype=np.float32)

    with torch.inference_mode():
        for i in range(depth):
            raw_slice = scan[:, :, i]
            if raw_slice.max() < 1e-6:
                continue
            norm_slice = normalize_slice(raw_slice)
            if norm_slice is None:
                continue

            img_128 = to_tensor_batch(norm_slice, IMG_SIZE, DEVICE)
            prob = F.softmax(classifier(img_128), dim=1)[0, 1].item()

            if prob <= CHOSEN_THRESHOLD:
                continue

            seg_pred = torch.sigmoid(segmenter(img_128))[0, 0]
            seg_pred_full = F.interpolate(
                seg_pred[None, None, ...], size=raw_slice.shape,
                mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            pred_stack[:, :, i] = (seg_pred_full > 0.5).astype(np.float32)

    return clean_small_fragments(pred_stack, min_size=min_fragment_size)


def compare_one_patient(flair_path, seg_path, patient_id, save_dir=OUTPUT_DIR, min_fragment_size=MIN_FRAGMENT_SIZE):
    scan = nib.load(flair_path).get_fdata()
    pred_stack = predict_full_volume(scan, min_fragment_size=min_fragment_size)

    has_ground_truth = seg_path is not None and os.path.exists(seg_path)
    true_stack = None
    volume_dice = None

    if has_ground_truth:
        true_stack = (nib.load(seg_path).get_fdata() > 0).astype(np.float32)
        intersection = (pred_stack * true_stack).sum()
        volume_dice = (2. * intersection + 1e-6) / (pred_stack.sum() + true_stack.sum() + 1e-6)

    pred_density = np.sum(pred_stack, axis=2)
    pred_density = pred_density / pred_density.max() if pred_density.max() > 0 else pred_density
    mip_brain = np.max(scan, axis=2)

    if has_ground_truth:
        true_density = np.sum(true_stack, axis=2)
        true_density = true_density / true_density.max() if true_density.max() > 0 else true_density

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        axes[0].imshow(mip_brain.T, cmap="gray", origin="lower")
        m0 = np.ma.masked_where(true_density.T == 0, true_density.T)
        axes[0].imshow(m0, cmap="cool", alpha=0.75, origin="lower", vmin=0, vmax=1)
        axes[0].set_title("GROUND TRUTH")
        axes[0].axis("off")

        axes[1].imshow(mip_brain.T, cmap="gray", origin="lower")
        m1 = np.ma.masked_where(pred_density.T == 0, pred_density.T)
        axes[1].imshow(m1, cmap="cool", alpha=0.75, origin="lower", vmin=0, vmax=1)
        axes[1].set_title("MODEL PREDICTION")
        axes[1].axis("off")

        fig.suptitle(f"{patient_id} -- Volume Dice: {volume_dice:.3f}", fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mip_brain.T, cmap="gray", origin="lower")
        m1 = np.ma.masked_where(pred_density.T == 0, pred_density.T)
        ax.imshow(m1, cmap="cool", alpha=0.75, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{patient_id} -- MODEL PREDICTION")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{patient_id}_comparison.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"{patient_id}: {'Dice='+format(volume_dice,'.3f') if has_ground_truth else 'no ground truth'} -- saved -> {out_path}")

    return volume_dice


if __name__ == "__main__":
    test_patients = DEFAULT_TEST_PATIENTS
    dice_scores = []

    for pid in test_patients:
        flair_path = os.path.join(BRATS_DIR, pid, f"{pid}_flair.nii")
        seg_path = os.path.join(BRATS_DIR, pid, f"{pid}_seg.nii")
        if not os.path.exists(flair_path):
            continue
        d = compare_one_patient(flair_path, seg_path, pid, min_fragment_size=MIN_FRAGMENT_SIZE)
        if d is not None:
            dice_scores.append(d)

    if len(dice_scores) > 0:
        print(f"\nMean volume Dice across {len(dice_scores)} patients: {np.mean(dice_scores):.4f}")