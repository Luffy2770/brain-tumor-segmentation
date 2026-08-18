import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classification"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "segmentation"))

from config import (
    BRATS_DIR, PROCESSED_DIR, OUTPUT_DIR, REPORT_DIR, DEVICE, IMG_SIZE, 
    CHOSEN_THRESHOLD, MIN_FRAGMENT_SIZE, CLASSIFIER_MODEL_PATH, UNET_MODEL_PATH,
    DEFAULT_PATIENT_ID
)
from model import TumorClassifierCNN
from unet_model import UNet2D
from utils import (
    normalize_slice, clean_small_fragments, to_tensor_batch,
    get_tumor_span_stats, generate_scan_timeline, load_segmenter_model
)

classifier = TumorClassifierCNN().to(DEVICE)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
classifier.eval()

segmenter = load_segmenter_model(DEVICE)
segmenter.eval()


def run_pipeline(flair_path, patient_id="patient", verbose=True, min_fragment_size=MIN_FRAGMENT_SIZE):
    scan = nib.load(flair_path).get_fdata()
    depth = scan.shape[2]
    print(f"[1/5] Loaded {patient_id}: shape {scan.shape}")

    full_mask_stack = np.zeros_like(scan, dtype=np.float32)
    report_rows = []
    slice_probs = []

    print(f"[2/5] Running classification + cascade segmentation on {depth} slices...")
    with torch.inference_mode():
        for i in range(depth):
            raw_slice = scan[:, :, i]

            if raw_slice.max() < 1e-6:
                report_rows.append({"slice": i, "classifier_prob": None, "verdict": "empty/skipped", "tumor_pixels": 0})
                slice_probs.append(0.0)
                continue

            norm_slice = normalize_slice(raw_slice)
            img_128 = to_tensor_batch(norm_slice, IMG_SIZE, DEVICE)
            prob = F.softmax(classifier(img_128), dim=1)[0, 1].item()
            slice_probs.append(prob)

            verdict = "TUMOR" if prob > CHOSEN_THRESHOLD else "no tumor"
            tumor_pixel_count = 0

            if prob > CHOSEN_THRESHOLD:
                seg_pred = torch.sigmoid(segmenter(img_128))[0, 0]
                seg_pred_full = F.interpolate(
                    seg_pred[None, None, ...], size=raw_slice.shape,
                    mode='bilinear', align_corners=False
                ).squeeze().cpu().numpy()
                binary_seg = (seg_pred_full > 0.5).astype(np.float32)
                full_mask_stack[:, :, i] = binary_seg
                tumor_pixel_count = int(binary_seg.sum())

            report_rows.append({
                "slice": i, "classifier_prob": round(prob, 4),
                "verdict": verdict, "tumor_pixels": tumor_pixel_count
            })

            if verbose and i % 25 == 0:
                print(f"   ...slice {i}/{depth} processed")

    print(f"[3/5] Cleaning small disconnected fragments (min_size={min_fragment_size})...")
    full_mask_stack = clean_small_fragments(full_mask_stack, min_size=min_fragment_size)

    for i in range(depth):
        report_rows[i]["tumor_pixels"] = int(full_mask_stack[:, :, i].sum())
        report_rows[i]["verdict"] = "TUMOR" if report_rows[i]["tumor_pixels"] > 0 else "no tumor"

    df = pd.DataFrame(report_rows)
    report_path = os.path.join(REPORT_DIR, f"{patient_id}_slice_report.csv")
    df.to_csv(report_path, index=False)
    print(f"[4/5] Saved per-slice report -> {report_path}")

    stats = get_tumor_span_stats(slice_probs, CHOSEN_THRESHOLD)
    print(f"\n--- Diagnostic Summary for {patient_id} ---")
    print(f"   Total Slices Analyzed: {stats['total_slices']}")
    print(f"   Tumor Flagged Slices : {stats['flagged_count']} ({stats['height_pct']}% of volume height)")
    if stats['flagged_count'] > 0:
        print(f"   Axial Tumor Range    : Slice #{stats['start_slice']} to #{stats['end_slice']}")
        print(f"   Peak Intensity Slice : Slice #{stats['peak_slice']} (Confidence: {stats['peak_prob']*100:.1f}%)")

    barcode_path = os.path.join(OUTPUT_DIR, f"{patient_id}_timeline_barcode.png")
    generate_scan_timeline(slice_probs, CHOSEN_THRESHOLD, barcode_path)
    print(f"   Saved 3D scan timeline barcode -> {barcode_path}")

    density_map = np.sum(full_mask_stack, axis=2)
    if density_map.max() > 0:
        density_map = density_map / density_map.max()

    mip_brain = np.max(scan, axis=2)

    print(f"\n[5/5] Rendering combined overlay...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mip_brain.T, cmap="gray", origin="lower")
    masked_density = np.ma.masked_where(density_map.T == 0, density_map.T)
    im = ax.imshow(masked_density, cmap="cool", alpha=0.75, origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Relative tumor presence across slices", fraction=0.046)
    ax.set_title(f"Combined Tumor Segmentation -- {patient_id}\n{stats['flagged_count']}/{depth} slices flagged ({stats['height_pct']}% span)")
    ax.axis("off")

    out_path = os.path.join(OUTPUT_DIR, f"{patient_id}_combined_overlay.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined overlay -> {out_path}")

    return df, full_mask_stack


if __name__ == "__main__":
    patient_id = DEFAULT_PATIENT_ID
    flair_path = os.path.join(BRATS_DIR, patient_id, f"{patient_id}_flair.nii")
    run_pipeline(flair_path, patient_id, min_fragment_size=MIN_FRAGMENT_SIZE)