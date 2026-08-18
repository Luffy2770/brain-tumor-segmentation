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
    BRATS_DIR, OUTPUT_DIR, REPORT_DIR, DEVICE, IMG_SIZE, 
    CHOSEN_THRESHOLD, MIN_FRAGMENT_SIZE, CLASSIFIER_MODEL_PATH,
    DEFAULT_PATIENT_ID, DEFAULT_TEST_PATIENTS
)
from model import TumorClassifierCNN
from utils import (
    normalize_slice, clean_small_fragments, to_tensor_batch, 
    select_smart_peak_slices, load_segmenter_model
)

classifier = TumorClassifierCNN().to(DEVICE)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
classifier.eval()

segmenter = load_segmenter_model(DEVICE)
segmenter.eval()


def find_patients_with_masks(base_dir=BRATS_DIR):
    patients_with_masks = []
    if not os.path.exists(base_dir):
        return patients_with_masks

    for pid in sorted(os.listdir(base_dir)):
        pdir = os.path.join(base_dir, pid)
        if os.path.isdir(pdir):
            has_flair = any("flair" in f.lower() for f in os.listdir(pdir))
            has_seg = any("seg" in f.lower() for f in os.listdir(pdir))
            if has_flair and has_seg:
                patients_with_masks.append(pid)

    return patients_with_masks


def compute_mask_metrics(true_mask, pred_mask, eps=1e-6):
    t = (true_mask > 0).astype(np.float32)
    p = (pred_mask > 0).astype(np.float32)

    tp = (t * p).sum()
    fp = ((1 - t) * p).sum()
    fn = (t * (1 - p)).sum()

    dice = (2.0 * tp + eps) / (t.sum() + p.sum() + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "true_voxels": int(t.sum()),
        "pred_voxels": int(p.sum())
    }


def compare_patient_masks(patient_id=DEFAULT_PATIENT_ID, base_dir=BRATS_DIR, save_dir=OUTPUT_DIR):
    patient_dir = os.path.join(base_dir, patient_id)
    flair_path = os.path.join(patient_dir, f"{patient_id}_flair.nii")
    seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")

    if not os.path.exists(flair_path) or not os.path.exists(seg_path):
        raise FileNotFoundError(f"Missing required MRI or ground-truth mask files for {patient_id}")

    scan = nib.load(flair_path).get_fdata()
    true_stack = (nib.load(seg_path).get_fdata() > 0).astype(np.float32)
    depth = scan.shape[2]

    pred_stack = np.zeros_like(scan, dtype=np.float32)
    slice_probs = []
    per_slice_reports = []

    with torch.inference_mode():
        for i in range(depth):
            raw_slice = scan[:, :, i]
            true_slice = true_stack[:, :, i]

            if raw_slice.max() < 1e-6:
                slice_probs.append(0.0)
                per_slice_reports.append({
                    "slice": i, "prob": 0.0, "true_pixels": int(true_slice.sum()),
                    "pred_pixels": 0, "slice_dice": 1.0 if true_slice.sum() == 0 else 0.0
                })
                continue

            norm_slice = normalize_slice(raw_slice)
            if norm_slice is None:
                slice_probs.append(0.0)
                per_slice_reports.append({
                    "slice": i, "prob": 0.0, "true_pixels": int(true_slice.sum()),
                    "pred_pixels": 0, "slice_dice": 1.0 if true_slice.sum() == 0 else 0.0
                })
                continue

            img_128 = to_tensor_batch(norm_slice, IMG_SIZE, DEVICE)
            prob = F.softmax(classifier(img_128), dim=1)[0, 1].item()
            slice_probs.append(prob)

            if prob > CHOSEN_THRESHOLD:
                seg_pred = torch.sigmoid(segmenter(img_128))[0, 0]
                seg_pred_full = F.interpolate(
                    seg_pred[None, None, ...], size=raw_slice.shape,
                    mode='bilinear', align_corners=False
                ).squeeze().cpu().numpy()
                binary_seg = (seg_pred_full > 0.5).astype(np.float32)
                pred_stack[:, :, i] = binary_seg

    pred_stack = clean_small_fragments(pred_stack, min_size=MIN_FRAGMENT_SIZE)

    for i in range(depth):
        t_sum = int(true_stack[:, :, i].sum())
        p_sum = int(pred_stack[:, :, i].sum())
        inter = int((true_stack[:, :, i] * pred_stack[:, :, i]).sum())
        s_dice = (2.0 * inter + 1e-6) / (t_sum + p_sum + 1e-6) if (t_sum + p_sum) > 0 else 1.0

        if i < len(per_slice_reports):
            per_slice_reports[i]["pred_pixels"] = p_sum
            per_slice_reports[i]["slice_dice"] = round(float(s_dice), 4)
        else:
            per_slice_reports.append({
                "slice": i, "prob": round(slice_probs[i], 4),
                "true_pixels": t_sum, "pred_pixels": p_sum,
                "slice_dice": round(float(s_dice), 4)
            })

    df_report = pd.DataFrame(per_slice_reports)
    report_file = os.path.join(REPORT_DIR, f"{patient_id}_mask_comparison.csv")
    df_report.to_csv(report_file, index=False)

    metrics = compute_mask_metrics(true_stack, pred_stack)

    print(f"\n=======================================================")
    print(f"       MASK COMPARISON REPORT: {patient_id}")
    print(f"=======================================================")
    print(f" 3D Volume Dice Score : {metrics['dice']:.4f}")
    print(f" 3D Volume IoU Score  : {metrics['iou']:.4f}")
    print(f" Precision (PPV)      : {metrics['precision']:.4f}")
    print(f" Recall (Sensitivity) : {metrics['recall']:.4f}")
    print(f" Ground Truth Voxels  : {metrics['true_voxels']} voxels")
    print(f" Predicted Voxels     : {metrics['pred_voxels']} voxels")
    print(f"=======================================================\n")

    mip_brain = np.max(scan, axis=2)
    true_density = np.sum(true_stack, axis=2)
    pred_density = np.sum(pred_stack, axis=2)

    tp_map = (true_stack * pred_stack).sum(axis=2)
    fp_map = ((1 - true_stack) * pred_stack).sum(axis=2)
    fn_map = (true_stack * (1 - pred_stack)).sum(axis=2)

    error_rgb = np.zeros((*mip_brain.shape, 3), dtype=np.float32)
    if tp_map.max() > 0: error_rgb[:, :, 1] = tp_map / tp_map.max()
    if fp_map.max() > 0: error_rgb[:, :, 0] = fp_map / fp_map.max()
    if fn_map.max() > 0: error_rgb[:, :, 2] = fn_map / fn_map.max()

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    axes[0, 0].imshow(mip_brain.T, cmap="gray", origin="lower")
    axes[0, 0].set_title(f"FLAIR MRI Scan ({patient_id})", fontsize=11, fontweight='bold')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mip_brain.T, cmap="gray", origin="lower")
    m0 = np.ma.masked_where(true_density.T == 0, true_density.T)
    axes[0, 1].imshow(m0, cmap="cool", alpha=0.75, origin="lower")
    axes[0, 1].set_title("Ground Truth Mask", fontsize=11, fontweight='bold')
    axes[0, 1].axis("off")

    axes[1, 0].imshow(mip_brain.T, cmap="gray", origin="lower")
    m1 = np.ma.masked_where(pred_density.T == 0, pred_density.T)
    axes[1, 0].imshow(m1, cmap="cool", alpha=0.75, origin="lower")
    axes[1, 0].set_title(f"Model Prediction (Dice: {metrics['dice']:.3f})", fontsize=11, fontweight='bold')
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mip_brain.T, cmap="gray", origin="lower")
    err_transposed = np.transpose(error_rgb, (1, 0, 2))[::-1, :]
    axes[1, 1].imshow(err_transposed, alpha=0.85, origin="lower")
    axes[1, 1].set_title("Discrepancy Map\n(Green=Match, Red=False Pos, Blue=False Neg)", fontsize=10, fontweight='bold')
    axes[1, 1].axis("off")

    fig.suptitle(f"Mask Comparison Summary -- {patient_id}\nDice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_img = os.path.join(save_dir, f"{patient_id}_mask_comparison_overview.png")
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sample_indices = select_smart_peak_slices(scan, np.array(slice_probs), (np.array(slice_probs) > CHOSEN_THRESHOLD).astype(int), num_samples=4)
    fig_grid, axes_grid = plt.subplots(len(sample_indices), 4, figsize=(13, 3.2 * len(sample_indices)))

    for row, idx in enumerate(sample_indices):
        raw_slc = scan[:, :, idx]
        gt_slc = true_stack[:, :, idx]
        pr_slc = pred_stack[:, :, idx]

        inter = (gt_slc * pr_slc).sum()
        slc_d = (2.0 * inter + 1e-6) / (gt_slc.sum() + pr_slc.sum() + 1e-6) if (gt_slc.sum() + pr_slc.sum()) > 0 else 1.0

        axes_grid[row, 0].imshow(raw_slc, cmap="gray")
        axes_grid[row, 0].set_title(f"Slice #{idx} -- FLAIR Input", fontsize=10)
        axes_grid[row, 0].axis("off")

        axes_grid[row, 1].imshow(raw_slc, cmap="gray")
        m_gt = np.ma.masked_where(gt_slc == 0, gt_slc)
        axes_grid[row, 1].imshow(m_gt, cmap="Reds", alpha=0.7)
        axes_grid[row, 1].set_title("Ground Truth Mask", fontsize=10)
        axes_grid[row, 1].axis("off")

        axes_grid[row, 2].imshow(raw_slc, cmap="gray")
        m_pr = np.ma.masked_where(pr_slc == 0, pr_slc)
        axes_grid[row, 2].imshow(m_pr, cmap="cool", alpha=0.7)
        axes_grid[row, 2].set_title(f"Model Mask (Dice: {slc_d:.2f})", fontsize=10)
        axes_grid[row, 2].axis("off")

        overlap = np.zeros((*raw_slc.shape, 3), dtype=np.float32)
        overlap[:, :, 1] = gt_slc * pr_slc
        overlap[:, :, 0] = (1 - gt_slc) * pr_slc
        overlap[:, :, 2] = gt_slc * (1 - pr_slc)

        axes_grid[row, 3].imshow(raw_slc, cmap="gray")
        axes_grid[row, 3].imshow(overlap, alpha=0.75)
        axes_grid[row, 3].set_title("Overlap & Discrepancy", fontsize=10)
        axes_grid[row, 3].axis("off")

    fig_grid.suptitle(f"Slice-by-Slice Mask Inspection Grid -- {patient_id}", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_grid = os.path.join(save_dir, f"{patient_id}_mask_slice_inspection.png")
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)

    return metrics, df_report


if __name__ == "__main__":
    patients_with_masks = find_patients_with_masks(BRATS_DIR)
    if len(patients_with_masks) > 0:
        target_patients = [p for p in DEFAULT_TEST_PATIENTS if p in patients_with_masks]
        if not target_patients:
            target_patients = patients_with_masks[:2]

        all_results = []
        for pid in target_patients:
            m, _ = compare_patient_masks(pid, base_dir=BRATS_DIR)
            m["patient_id"] = pid
            all_results.append(m)

        df_summary = pd.DataFrame(all_results)
        print("\n=======================================================")
        print("         SUMMARY ACROSS EVALUATED PATIENTS")
        print("=======================================================")
        print(df_summary[["patient_id", "dice", "iou", "precision", "recall", "true_voxels", "pred_voxels"]])
        print(f"\nMean Volume Dice: {df_summary['dice'].mean():.4f}")
        print(f"Mean Volume IoU : {df_summary['iou'].mean():.4f}")
        print("=======================================================")
