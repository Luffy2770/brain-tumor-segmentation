import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BRATS_DIR, PROCESSED_DIR, DEVICE, CHOSEN_THRESHOLD, CLASSIFIER_MODEL_PATH, DEFAULT_PATIENT_ID, IMG_SIZE
from classification.model import TumorClassifierCNN
from utils import (
    normalize_slice, to_tensor_batch, generate_scan_timeline, 
    select_smart_peak_slices, get_tumor_span_stats
)


def run_inference_v1(patient_id=DEFAULT_PATIENT_ID):
    patient_dir = os.path.join(BRATS_DIR, patient_id)
    if not os.path.exists(patient_dir):
        print(f"Patient directory not found: {patient_dir}")
        return None, None

    flair_candidates = [f for f in os.listdir(patient_dir) if "flair" in f.lower()]
    if not flair_candidates:
        raise FileNotFoundError(f"No flair file found in {patient_dir}")
    flair_path = os.path.join(patient_dir, flair_candidates[0])

    model = TumorClassifierCNN().to(DEVICE)
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    model.eval()

    scan = nib.load(flair_path).get_fdata()
    depth = scan.shape[2]

    slice_probs = []
    with torch.inference_mode():
        for i in range(depth):
            raw_slice = scan[:, :, i]
            if raw_slice.max() < 1e-6:
                slice_probs.append(0.0)
                continue

            norm_slice = normalize_slice(raw_slice)
            if norm_slice is None:
                slice_probs.append(0.0)
                continue

            img_128 = to_tensor_batch(norm_slice, IMG_SIZE, DEVICE)
            prob = F.softmax(model(img_128), dim=1)[0, 1].item()
            slice_probs.append(prob)

    slice_probs = np.array(slice_probs)
    predictions = (slice_probs > CHOSEN_THRESHOLD).astype(int)

    stats = get_tumor_span_stats(slice_probs, CHOSEN_THRESHOLD)
    print(f"\n--- Prediction Summary for {patient_id} ---")
    print(f"Total slices: {stats['total_slices']}")
    print(f"Slices flagged as TUMOR: {stats['flagged_count']} ({stats['height_pct']}% of height)")
    if stats['flagged_count'] > 0:
        print(f"Tumor-flagged slice range: Slice #{stats['start_slice']} to #{stats['end_slice']}")
        print(f"Peak Intensity Slice: Slice #{stats['peak_slice']} (Confidence: {stats['peak_prob']*100:.1f}%)")

    barcode_path = os.path.join(PROCESSED_DIR, f"{patient_id}_timeline_barcode.png")
    generate_scan_timeline(slice_probs, CHOSEN_THRESHOLD, barcode_path)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(slice_probs, color='crimson')
    ax1.axhline(CHOSEN_THRESHOLD, color='gray', linestyle='--', label=f"threshold={CHOSEN_THRESHOLD}")
    ax1.fill_between(range(len(slice_probs)), slice_probs, CHOSEN_THRESHOLD,
                      where=(slice_probs > CHOSEN_THRESHOLD), color='crimson', alpha=0.2)
    ax1.set_xlabel("Slice index (Inferior -> Superior)")
    ax1.set_ylabel("Predicted tumor probability")
    ax1.set_title(f"Tumor Probability Across Full Scan -- {patient_id}")
    ax1.legend()
    plt.tight_layout()
    prof_path = os.path.join(PROCESSED_DIR, f"{patient_id}_probability_profile.png")
    plt.savefig(prof_path, dpi=140)
    plt.close(fig1)

    sample_indices = select_smart_peak_slices(scan, slice_probs, predictions, num_samples=8)
    fig2, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, idx in zip(axes.flat, sample_indices):
        ax.imshow(scan[:, :, idx], cmap="gray")
        label = "TUMOR" if predictions[idx] == 1 else "no tumor"
        color = "red" if predictions[idx] == 1 else "green"
        ax.set_title(f"Slice {idx}\n{label} ({slice_probs[idx]:.2f})", color=color, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"Smart Peak Slice Predictions -- {patient_id}")
    plt.tight_layout()
    grid_path = os.path.join(PROCESSED_DIR, f"{patient_id}_sample_grid.png")
    plt.savefig(grid_path, dpi=140)
    plt.close(fig2)

    return slice_probs, predictions


if __name__ == "__main__":
    run_inference_v1()