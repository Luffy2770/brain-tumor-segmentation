import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DiagnosticPackager:
    """
    Station 8: Clean & Simple Output Packager.
    Generates:
    1. 2D Top-Down Stacked Overview Image (prediction.png or comparison.png)
    2. Slice Audit CSV (slice_report.csv)
    """
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_patient_dir(self, patient_id):
        p_dir = os.path.join(self.output_dir, patient_id)
        os.makedirs(p_dir, exist_ok=True)
        return p_dir

    def save_prediction_image(self, patient_id, raw_3d_scan, pred_mask_3d):
        """
        MODE 1: Predict Only. 2D top-down stacked brain view with predicted tumor overlay.
        """
        p_dir = self._get_patient_dir(patient_id)
        mip_brain = np.max(raw_3d_scan, axis=2)
        tumor_stack = np.sum(pred_mask_3d > 0, axis=2)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mip_brain.T, cmap="gray", origin="lower")
        if np.max(tumor_stack) > 0:
            m_pred = np.ma.masked_where(tumor_stack.T == 0, tumor_stack.T)
            im = ax.imshow(m_pred, cmap="autumn", alpha=0.7, origin="lower")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label("Tumor thickness (slices)", fontsize=10, fontweight="bold")

        total_voxels = int(pred_mask_3d.sum())
        ax.set_title(f"Tumor Prediction -- {patient_id}\n({total_voxels:,} voxels | {total_voxels/1000.0:.1f} mL)", fontsize=12, fontweight='bold')
        ax.axis("off")

        out_path = os.path.abspath(os.path.join(p_dir, "prediction.png"))
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Station 8] Saved Prediction -> {out_path}")
        return out_path

    def save_comparison_image(self, patient_id, raw_3d_scan, true_mask_3d, pred_mask_3d, dice_score=None):
        """
        MODE 2: Compare Mode. 3 Panels [1. Normal Scan, 2. Real Mask, 3. Predicted Mask].
        Top-down stacked view across all slices with strictly shared color scaling.
        """
        p_dir = self._get_patient_dir(patient_id)
        mip_brain = np.max(raw_3d_scan, axis=2)
        true_stack = np.sum(true_mask_3d > 0, axis=2)
        pred_stack = np.sum(pred_mask_3d > 0, axis=2)

        true_ml = np.sum(true_mask_3d > 0) / 1000.0
        pred_ml = np.sum(pred_mask_3d > 0) / 1000.0
        max_thickness = max(int(np.max(true_stack)), int(np.max(pred_stack)), 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

        # Panel 1: Normal Scan (Top-down view)
        axes[0].imshow(mip_brain.T, cmap="gray", origin="lower")
        axes[0].set_title(f"1. Normal Scan\nPatient: {patient_id}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Real Mask (Ground Truth top-down view)
        axes[1].imshow(mip_brain.T, cmap="gray", origin="lower")
        im_true = None
        if np.max(true_stack) > 0:
            m_true = np.ma.masked_where(true_stack.T == 0, true_stack.T)
            im_true = axes[1].imshow(m_true, cmap="autumn", vmin=1, vmax=max_thickness, alpha=0.7, origin="lower")
        axes[1].set_title(f"2. Real Mask (Ground Truth)\n({true_ml:.1f} mL | {int(np.sum(true_mask_3d)):,} voxels)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Predicted Mask (Top-down view)
        axes[2].imshow(mip_brain.T, cmap="gray", origin="lower")
        im_pred = None
        if np.max(pred_stack) > 0:
            m_pred = np.ma.masked_where(pred_stack.T == 0, pred_stack.T)
            im_pred = axes[2].imshow(m_pred, cmap="autumn", vmin=1, vmax=max_thickness, alpha=0.7, origin="lower")
        dice_text = f"Dice: {dice_score*100:.1f}% | " if dice_score is not None else ""
        axes[2].set_title(f"3. Predicted Mask\n({dice_text}{pred_ml:.1f} mL | {int(np.sum(pred_mask_3d)):,} voxels)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Shared colorbar on the rightmost panel with common thickness scale
        active_im = im_pred if im_pred is not None else im_true
        if active_im is not None:
            cbar = fig.colorbar(active_im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Tumor thickness (slices)", fontsize=10, fontweight="bold")

        dice_header = f" -- Dice: {dice_score*100:.1f}%" if dice_score is not None else ""
        fig.suptitle(f"Top-Down Tumor Stack Comparison -- {patient_id}{dice_header}", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_path = os.path.abspath(os.path.join(p_dir, "comparison.png"))
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Station 8] Saved Comparison -> {out_path}")
        return out_path

    def save_report_csv(self, patient_id, report_rows, pred_mask_3d):
        p_dir = self._get_patient_dir(patient_id)
        depth = pred_mask_3d.shape[2]
        for i in range(depth):
            pixels = int(pred_mask_3d[:, :, i].sum())
            report_rows[i]["tumor_pixels"] = pixels
            report_rows[i]["verdict"] = "TUMOR" if pixels > 0 else "no tumor"

        df = pd.DataFrame(report_rows)
        csv_path = os.path.join(p_dir, "slice_report.csv")
        df.to_csv(csv_path, index=False)
        return df

    def save_slice_distribution_bar(self, patient_id, pred_mask_3d, true_mask_3d=None):
        """
        Generates a per-slice bar graph showing tumor voxel count across all 155 slices.
        """
        p_dir = self._get_patient_dir(patient_id)
        depth = pred_mask_3d.shape[2]
        slice_indices = np.arange(depth)
        pred_counts = np.sum(pred_mask_3d > 0, axis=(0, 1))

        fig, ax = plt.subplots(figsize=(12, 4.5))

        if true_mask_3d is not None:
            true_counts = np.sum(true_mask_3d > 0, axis=(0, 1))
            ax.bar(slice_indices - 0.2, true_counts, width=0.4, color='#1f77b4', alpha=0.85, label='Ground Truth')
            ax.bar(slice_indices + 0.2, pred_counts, width=0.4, color='#ff7f0e', alpha=0.85, label='Predicted')
            ax.legend(fontsize=10)
            ax.set_title(f"Per-Slice Tumor Voxel Count -- {patient_id} (Ground Truth vs Prediction)", fontsize=12, fontweight='bold')
        else:
            ax.bar(slice_indices, pred_counts, width=0.7, color='#ff7f0e', alpha=0.85, label='Predicted Tumor')
            ax.legend(fontsize=10)
            ax.set_title(f"Per-Slice Tumor Voxel Count -- {patient_id} (Predicted)", fontsize=12, fontweight='bold')

        ax.set_xlabel("Brain Slice Index (0 = Neck to 154 = Top of Head)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Tumor Voxel Count", fontsize=10, fontweight='bold')
        ax.set_xlim(-1, depth)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        out_path = os.path.abspath(os.path.join(p_dir, "slice_distribution.png"))
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Station 8] Saved Slice Distribution Bar Graph -> {out_path}")
        return out_path