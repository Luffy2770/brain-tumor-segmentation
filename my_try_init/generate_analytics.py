import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import nibabel as nib

from config import (
    DATASET_DIR, DATA_DIR, OUTPUTS_DIR, DEVICE,
    CLASSIFIER_WEIGHTS, CARVER_WEIGHTS, CLASSIFIER_THRESHOLD,
    SEG_BINARY_THRESHOLD, MIN_FRAGMENT_SIZE
)
from s1 import ScanLoader
from s2 import SliceNormalizer
from s3 import TensorFunnel
from s4 import SorterGate
from s5 import PrecisionCarver
from s6 import ResolutionRestorer
from s7 import VolumetricSieve
from factory_pipeline import predict_volume

def generate_all_analytics():
    """
    Generates the exact 3 core evaluation graphs for presentation:
    1. Test Accuracy & Metrics on 37 Unseen Patients (test_accuracy_37_patients.png)
    2. Stage 4 Classifier Slice Confusion Matrix (stage4_classifier_confusion_matrix.png)
    3. Stage 5 Attention U-Net Voxel Confusion Matrix (stage5_segmenter_confusion_matrix.png)
    """
    print("============================================================")
    print("[ANALYTICS] GENERATING THE 3 MASTER EVALUATION GRAPHS")
    print("============================================================")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # 1. GRAPH 1: ACCURACY ON 37 UNSEEN CASES
    # -------------------------------------------------------------
    summary_csv = os.path.join(OUTPUTS_DIR, "all_test_patients_benchmark.csv")
    if not os.path.exists(summary_csv):
        from factory_pipeline import run_all_test_patients
        run_all_test_patients()

    df = pd.read_csv(summary_csv)
    n_pids = len(df)
    mean_dice = df['dice'].mean()
    median_dice = df['dice'].median()
    std_dice = df['dice'].std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left: Sorted Patient Dice Curve
    df_sorted = df.sort_values(by="dice").reset_index(drop=True)
    ax1.plot(df_sorted.index + 1, df_sorted["dice"], color="#1f77b4", linewidth=2.2, marker='o', markersize=4.5, label="3D Dice Score")
    ax1.axhline(mean_dice, color="crimson", linestyle="--", linewidth=1.8, label=f"Mean: {mean_dice*100:.1f}% ± {std_dice*100:.1f}%")
    ax1.axhline(median_dice, color="forestgreen", linestyle=":", linewidth=1.8, label=f"Median: {median_dice*100:.1f}%")
    ax1.fill_between(df_sorted.index + 1, mean_dice - std_dice, mean_dice + std_dice, color="crimson", alpha=0.1)

    ax1.set_title(f"3D Segmentation Accuracy (n = {n_pids} Unseen Test Patients)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Patients (Ranked Lowest to Highest)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("3D Volume Dice Score", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlim(0.5, n_pids + 0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # Right: Metric Spread Boxplot
    metrics = ["dice", "recall", "precision"]
    df_melted = df.melt(id_vars=["patient_id"], value_vars=metrics, var_name="Metric", value_name="Score")
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    sns.boxplot(data=df_melted, x="Metric", y="Score", hue="Metric", palette=palette, ax=ax2, width=0.38, boxprops=dict(alpha=0.75), legend=False)
    sns.stripplot(data=df_melted, x="Metric", y="Score", color="black", size=5.5, jitter=0.18, alpha=0.75, ax=ax2)

    ax2.set_title(f"Overall Metric Spread (n = {n_pids} Patients)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Evaluation Metric", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Score (0.0 to 1.0)", fontsize=10, fontweight="bold")
    ax2.set_xticklabels(["3D Dice", "Recall (Sens.)", "Precision (PPV)"], fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)

    plt.tight_layout()
    g1_path = os.path.join(OUTPUTS_DIR, "test_accuracy_37_patients.png")
    plt.savefig(g1_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Graph 1] Saved Test Accuracy Graph -> {g1_path}")

    # -------------------------------------------------------------
    # 2 & 3: EVALUATE STAGE 4 & STAGE 5 MATRICES ON 37 TEST CASES
    # -------------------------------------------------------------
    test_pids = sorted(list(set(np.load(os.path.join(DATA_DIR, "seg_test_pids.npy")))))

    # Initialize Shared Pipeline Engines
    sorter     = SorterGate(weights_path=CLASSIFIER_WEIGHTS, threshold=CLASSIFIER_THRESHOLD, device=DEVICE)
    carver     = PrecisionCarver(weights_path=CARVER_WEIGHTS, device=DEVICE)
    normalizer = SliceNormalizer()
    funnel     = TensorFunnel(target_size=128, device=DEVICE)
    restorer   = ResolutionRestorer(binary_threshold=SEG_BINARY_THRESHOLD)
    sieve      = VolumetricSieve(min_size=MIN_FRAGMENT_SIZE)

    # Stage 4 Slice-level counters (non-empty brain slices)
    s4_tp, s4_fp, s4_tn, s4_fn = 0, 0, 0, 0

    # Stage 5 Voxel-level counters across all test volumes
    vox_tp, vox_fp, vox_tn, vox_fn = 0, 0, 0, 0

    with torch.inference_mode():
        for pid in test_pids:
            pdir = os.path.join(DATASET_DIR, pid)
            flair = os.path.join(pdir, f"{pid}_flair.nii")
            seg = os.path.join(pdir, f"{pid}_seg.nii")
            if not (os.path.exists(flair) and os.path.exists(seg)):
                continue

            loader = ScanLoader(flair)
            gt_vol = (nib.load(seg).get_fdata() > 0).astype(np.float32)

            # Call Single Shared Prediction Function (includes skull mask + 3D sieve)
            clean_pred_vol, report_rows = predict_volume(
                loader=loader,
                normalizer=normalizer,
                funnel=funnel,
                sorter=sorter,
                carver=carver,
                restorer=restorer,
                sieve=sieve
            )

            # Update Stage 4 Slice Matrix (only non-empty slices)
            for row in report_rows:
                if row.get("is_empty", False):
                    continue
                s = row["slice"]
                gt_slice = gt_vol[:, :, s]
                has_tumor = bool((gt_slice > 0).any())
                passed_to_carver = row["passed_to_carver"]

                if has_tumor and passed_to_carver:
                    s4_tp += 1
                elif (not has_tumor) and passed_to_carver:
                    s4_fp += 1
                elif (not has_tumor) and (not passed_to_carver):
                    s4_tn += 1
                elif has_tumor and (not passed_to_carver):
                    s4_fn += 1

            # Update Stage 5 Voxel Matrix
            tf = gt_vol.flatten()
            pf = clean_pred_vol.flatten()
            vox_tp += int(np.sum((pf == 1) & (tf == 1)))
            vox_fp += int(np.sum((pf == 1) & (tf == 0)))
            vox_fn += int(np.sum((pf == 0) & (tf == 1)))
            vox_tn += int(np.sum((pf == 0) & (tf == 0)))

    # -------------------------------------------------------------
    # GRAPH 2: STAGE 4 CLASSIFIER CONFUSION MATRIX (SLICES)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    s4_total = s4_tp + s4_fp + s4_tn + s4_fn
    cm_s4 = np.array([[s4_tn, s4_fp], [s4_fn, s4_tp]])
    cm_s4_pct = cm_s4 / s4_total * 100.0

    lbls_s4 = np.array([
        [f"True Neg (Healthy Filtered)\n{s4_tn:,} slices\n({cm_s4_pct[0,0]:.1f}%)",
         f"False Pos (Healthy Passed)\n{s4_fp:,} slices\n({cm_s4_pct[0,1]:.1f}%)"],
        [f"False Neg (Tumor Missed)\n{s4_fn:,} slices\n({cm_s4_pct[1,0]:.1f}%)",
         f"True Pos (Tumor Passed)\n{s4_tp:,} slices\n({cm_s4_pct[1,1]:.1f}%)"]
    ])

    sns.heatmap(cm_s4, annot=lbls_s4, fmt="", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"fontsize": 10, "fontweight": "bold"}, linewidths=1.5, linecolor="gray")

    s4_sens = s4_tp / max(1, s4_tp + s4_fn) * 100
    s4_spec = s4_tn / max(1, s4_tn + s4_fp) * 100
    ax.set_title(f"Stage 4 Classifier: Slice Confusion Matrix\n(Sensitivity: {s4_sens:.1f}% | Specificity: {s4_spec:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Slice Action (Stage 4)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Slice Category (Ground Truth)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(["Filtered (Healthy)", "Passed to Carver"], fontsize=10)
    ax.set_yticklabels(["Healthy Slice", "Tumor Slice"], fontsize=10, va='center')

    plt.tight_layout()
    g2_path = os.path.join(OUTPUTS_DIR, "stage4_classifier_confusion_matrix.png")
    plt.savefig(g2_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Graph 2] Saved Stage 4 Classifier Matrix -> {g2_path}")

    # -------------------------------------------------------------
    # GRAPH 3: STAGE 5 SEGMENTER CONFUSION MATRIX (VOXELS)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vox_total = vox_tp + vox_fp + vox_fn + vox_tn
    cm_s5 = np.array([[vox_tn, vox_fp], [vox_fn, vox_tp]])
    cm_s5_pct = cm_s5 / vox_total * 100.0

    lbls_s5 = np.array([
        [f"True Neg (Healthy Tissue)\n{vox_tn:,} voxels\n({cm_s5_pct[0,0]:.1f}%)",
         f"False Pos (Over-segmented)\n{vox_fp:,} voxels\n({cm_s5_pct[0,1]:.1f}%)"],
        [f"False Neg (Missed Tumor)\n{vox_fn:,} voxels\n({cm_s5_pct[1,0]:.1f}%)",
         f"True Pos (Segmented Tumor)\n{vox_tp:,} voxels\n({cm_s5_pct[1,1]:.1f}%)"]
    ])

    sns.heatmap(cm_s5, annot=lbls_s5, fmt="", cmap="Purples", cbar=False, ax=ax,
                annot_kws={"fontsize": 10, "fontweight": "bold"}, linewidths=1.5, linecolor="gray")

    vox_sens = vox_tp / max(1, vox_tp + vox_fn) * 100
    vox_prec = vox_tp / max(1, vox_tp + vox_fp) * 100
    ax.set_title(f"Stage 5 Attention U-Net: Voxel Confusion Matrix\n(Voxel Sensitivity: {vox_sens:.1f}% | Precision: {vox_prec:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Voxel Classification (Stage 5)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Voxel Category (Ground Truth)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(["Healthy Voxel", "Tumor Voxel"], fontsize=10)
    ax.set_yticklabels(["Healthy Voxel", "Tumor Voxel"], fontsize=10, va='center')

    plt.tight_layout()
    g3_path = os.path.join(OUTPUTS_DIR, "stage5_segmenter_confusion_matrix.png")
    plt.savefig(g3_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Graph 3] Saved Stage 5 Segmenter Matrix -> {g3_path}")

    print("\n[SUCCESS] All 3 master analytics graphs generated in outputs/!")

if __name__ == "__main__":
    generate_all_analytics()
