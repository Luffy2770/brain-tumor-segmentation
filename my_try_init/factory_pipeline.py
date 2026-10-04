import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
import pandas as pd

from config import (
    DATASET_DIR, PROJECT_ROOT, OUTPUTS_DIR, REPORTS_DIR, DEVICE,
    CLASSIFIER_WEIGHTS, CARVER_WEIGHTS, CLASSIFIER_THRESHOLD,
    SEG_BINARY_THRESHOLD, MIN_FRAGMENT_SIZE, SAMPLE_DATA_DIR, DATA_DIR
)
from s1 import ScanLoader
from s2 import SliceNormalizer
from s3 import TensorFunnel
from s4 import SorterGate
from s5 import PrecisionCarver
from s6 import ResolutionRestorer
from s7 import VolumetricSieve
from s8 import DiagnosticPackager

def find_patient_files(patient_target):
    """
    Auto-locates flair and seg files from patient ID, folder, or file path.
    """
    # 1. If direct file path
    if os.path.isfile(patient_target):
        flair_path = patient_target
        p_id = os.path.basename(flair_path).split(".")[0].replace("_flair", "")
        # Check if sibling seg file exists
        p_dir = os.path.dirname(flair_path)
        seg_candidates = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if "seg" in f.lower() and (f.endswith(".nii") or f.endswith(".nii.gz"))]
        seg_path = seg_candidates[0] if seg_candidates else None
        return p_id, flair_path, seg_path

    # 2. If directory path
    if os.path.isdir(patient_target):
        p_dir = os.path.normpath(patient_target)
        p_id = os.path.basename(p_dir)
        flairs = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if "flair" in f.lower() and (f.endswith(".nii") or f.endswith(".nii.gz"))]
        segs   = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if "seg" in f.lower() and (f.endswith(".nii") or f.endswith(".nii.gz"))]
        return p_id, flairs[0] if flairs else None, segs[0] if segs else None

    # 3. If patient ID string (e.g. 'BraTS20_Training_230' or 'BraTS20_Validation_001')
    p_id = patient_target.strip()
    search_dirs = [
        DATASET_DIR,
        os.path.join(PROJECT_ROOT, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData"),
        SAMPLE_DATA_DIR
    ]

    for base_d in search_dirs:
        if os.path.exists(base_d):
            p_dir = os.path.join(base_d, p_id)
            if os.path.isdir(p_dir):
                flairs = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if "flair" in f.lower() and (f.endswith(".nii") or f.endswith(".nii.gz"))]
                segs   = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if "seg" in f.lower() and (f.endswith(".nii") or f.endswith(".nii.gz"))]
                return p_id, flairs[0] if flairs else None, segs[0] if segs else None

    return p_id, None, None


def predict_volume(
    loader,
    normalizer,
    funnel,
    sorter,
    carver,
    restorer,
    sieve,
):
    """
    Unified end-to-end volumetric inference routine.
    Guarantees that pipeline execution, benchmarking, and analytics use the exact same
    normalization, thresholding, skull masking, and 3D sieving rules.
    """
    depth = loader.total_slices
    raw_shape = loader.get_slice(0).shape
    full_mask_stack = np.zeros((raw_shape[0], raw_shape[1], depth), dtype=np.float32)
    report_rows = []

    for i in range(depth):
        raw_slice = loader.get_slice(i)

        if loader.is_empty_slice(raw_slice):
            report_rows.append({
                "slice": i,
                "classifier_prob": 0.0,
                "verdict": "empty/skipped",
                "tumor_pixels": 0,
                "passed_to_carver": False,
                "is_empty": True
            })
            continue

        clean_slice = normalizer.normalize(raw_slice, mask_background=False)
        if clean_slice is None:
            report_rows.append({
                "slice": i,
                "classifier_prob": 0.0,
                "verdict": "empty/skipped",
                "tumor_pixels": 0,
                "passed_to_carver": False,
                "is_empty": True
            })
            continue

        tensor_crate = funnel.prepare_slice(clean_slice)
        prob, is_suspicious = sorter.inspect_slice(tensor_crate)

        tumor_pixels = 0
        if is_suspicious:
            prob_map_128 = carver.carve_mask(tensor_crate)
            final_mask_2d = restorer.restore_and_binarize(prob_map_128, target_shape=raw_slice.shape)
            full_mask_stack[:, :, i] = final_mask_2d
            tumor_pixels = int(final_mask_2d.sum())

        report_rows.append({
            "slice": i,
            "classifier_prob": round(float(prob), 4),
            "verdict": "TUMOR" if tumor_pixels > 0 else "no tumor",
            "tumor_pixels": tumor_pixels,
            "passed_to_carver": bool(is_suspicious),
            "is_empty": False
        })

    # Skull Masking: Anatomically impossible for tumor to exist in empty air outside skull
    full_mask_stack[loader.volume_data == 0] = 0.0

    # Station 7: 3D Sieve
    cleaned_3d_mask = sieve.clean_volume(full_mask_stack)

    return cleaned_3d_mask, report_rows


def run_contraption(
    flair_path,
    seg_path=None,
    patient_id=None,
    mode=None,
    output_dir=OUTPUTS_DIR,
    device=DEVICE,
    classifier_weights=CLASSIFIER_WEIGHTS,
    carver_weights=CARVER_WEIGHTS,
    classifier_threshold=CLASSIFIER_THRESHOLD,
    seg_binary_threshold=SEG_BINARY_THRESHOLD,
    min_fragment_size=MIN_FRAGMENT_SIZE,
    verbose=True
):
    """
    Executes the 8-Station Contraption Pipeline on a 3D MRI Scan.
    - Automatically sets mode='compare' if ground truth seg_path exists.
    - Otherwise runs mode='predict'.
    """
    if mode is None:
        mode = "compare" if (seg_path and os.path.exists(seg_path)) else "predict"

    if verbose:
        print(f"============================================================")
        print(f"[PIPELINE] Processing: {patient_id} (Mode: {mode.upper()})")
        print(f"   Device Engine: {device}")
        print(f"============================================================")

    # Initialize Station Machines
    loader     = ScanLoader(flair_path)
    normalizer = SliceNormalizer()
    funnel     = TensorFunnel(target_size=128, device=device)
    sorter     = SorterGate(weights_path=classifier_weights, threshold=classifier_threshold, device=device)
    carver     = PrecisionCarver(weights_path=carver_weights, device=device)
    restorer   = ResolutionRestorer(binary_threshold=seg_binary_threshold)
    sieve      = VolumetricSieve(min_size=min_fragment_size)
    packager   = DiagnosticPackager(output_dir=output_dir)

    # Unified Inference Execution
    cleaned_3d_mask, report_rows = predict_volume(
        loader=loader,
        normalizer=normalizer,
        funnel=funnel,
        sorter=sorter,
        carver=carver,
        restorer=restorer,
        sieve=sieve
    )

    # Save Slice Report CSV in patient folder
    packager.save_report_csv(patient_id, report_rows, cleaned_3d_mask)

    # MODE 1: PREDICT ONLY
    if mode == "predict" or seg_path is None or not os.path.exists(seg_path):
        packager.save_prediction_image(patient_id, loader.volume_data, cleaned_3d_mask)
        packager.save_slice_distribution_bar(patient_id, cleaned_3d_mask)
        total_voxels = int(cleaned_3d_mask.sum())
        if verbose:
            print(f"[RESULT] Finished Prediction for {patient_id}. Detected Tumor Voxels: {total_voxels:,}\n")
        return cleaned_3d_mask, None

    # MODE 2: COMPARE WITH GROUND TRUTH (3 PANELS)
    if mode == "compare" and seg_path and os.path.exists(seg_path):
        true_3d = (nib.load(seg_path).get_fdata() > 0).astype(np.float32)
        
        tp = (true_3d * cleaned_3d_mask).sum()
        fp = ((1 - true_3d) * cleaned_3d_mask).sum()
        fn = (true_3d * (1 - cleaned_3d_mask)).sum()
        eps = 1e-6
        
        dice = (2.0 * tp + eps) / (true_3d.sum() + cleaned_3d_mask.sum() + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        recall = (tp + eps) / (tp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)

        true_ml = float(true_3d.sum()) / 1000.0
        pred_ml = float(cleaned_3d_mask.sum()) / 1000.0

        metrics = {
            "patient_id": patient_id,
            "dice": float(dice),
            "iou": float(iou),
            "recall": float(recall),
            "precision": float(precision),
            "true_voxels": int(true_3d.sum()),
            "pred_voxels": int(cleaned_3d_mask.sum()),
            "true_ml": round(true_ml, 2),
            "pred_ml": round(pred_ml, 2)
        }

        if verbose:
            print(f"============================================================")
            print(f"       SCORECARD: {patient_id}")
            print(f"============================================================")
            print(f" 3D Volume Dice Score : {dice*100:.2f}%")
            print(f" 3D Volume IoU Score  : {iou*100:.2f}%")
            print(f" Recall (Sensitivity) : {recall*100:.2f}%")
            print(f" Precision (PPV)      : {precision*100:.2f}%")
            print(f" Ground Truth Volume  : {true_ml:.1f} mL ({int(true_3d.sum()):,} voxels)")
            print(f" Predicted Volume     : {pred_ml:.1f} mL ({int(cleaned_3d_mask.sum()):,} voxels)")
            print(f"============================================================\n")

        # Save 3-panel Top-Down Stack Comparison Image & Slice Distribution
        packager.save_comparison_image(patient_id, loader.volume_data, true_3d, cleaned_3d_mask, dice_score=dice)
        packager.save_slice_distribution_bar(patient_id, cleaned_3d_mask, true_mask_3d=true_3d)

        return cleaned_3d_mask, metrics


def run_all_test_patients():
    """
    Runs prediction & comparison on all 37 UNSEEN test patients from the dataset split.
    """
    test_pids_path = os.path.join(DATA_DIR, "seg_test_pids.npy")
    if not os.path.exists(test_pids_path):
        print("[ERROR] seg_test_pids.npy not found in data/ folder. Run prepare_data.py first!")
        return

    test_pids = sorted(list(set(np.load(test_pids_path))))
    print(f"============================================================")
    print(f"[BENCHMARK] RUNNING FULL TEST BENCHMARK ON ALL {len(test_pids)} UNSEEN PATIENTS")
    print(f"============================================================")

    all_results = []
    for idx, pid in enumerate(test_pids, start=1):
        _, flair, seg = find_patient_files(pid)
        if not flair:
            continue
        print(f"[{idx:02d}/{len(test_pids):02d}] Processing {pid}...")
        _, metrics = run_contraption(flair, seg_path=seg, patient_id=pid, mode="compare", verbose=False)
        if metrics:
            all_results.append(metrics)
            print(f"       -> Dice: {metrics['dice']*100:.2f}% | IoU: {metrics['iou']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | Precision: {metrics['precision']*100:.2f}%")

    if all_results:
        df = pd.DataFrame(all_results)
        summary_csv = os.path.join(OUTPUTS_DIR, "all_test_patients_benchmark.csv")
        df.to_csv(summary_csv, index=False)
        print(f"\n============================================================")
        print(f"       FINAL BENCHMARK SCORECARD ACROSS {len(df)} TEST PATIENTS")
        print(f"============================================================")
        print(f" Mean 3D Volume Dice  : {df['dice'].mean()*100:.2f}%")
        print(f" Mean 3D Volume IoU   : {df['iou'].mean()*100:.2f}%")
        print(f" Mean Recall / Sens.  : {df['recall'].mean()*100:.2f}%")
        print(f" Mean Precision (PPV) : {df['precision'].mean()*100:.2f}%")
        print(f" Saved full summary CSV -> {summary_csv}")
        print(f"============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8-Station Brain Tumor Segmentation Contraption")
    parser.add_argument("patient", nargs="?", default=None, help="Patient ID, folder, or scan file path")
    parser.add_argument("--test-all", action="store_true", help="Run benchmark across all 37 unseen test patients")
    parser.add_argument("--mode", "-m", choices=["predict", "compare"], default=None, help="Force predict or compare mode")

    args = parser.parse_args()

    if args.test_all:
        run_all_test_patients()
    elif args.patient:
        pid, flair, seg = find_patient_files(args.patient)
        if not flair:
            print(f"[ERROR] Could not find scan for: {args.patient}")
            sys.exit(1)
        run_contraption(flair, seg_path=seg, patient_id=pid, mode=args.mode)
    else:
        # Default sample run
        pid, flair, seg = find_patient_files("BraTS20_Training_001")
        if flair:
            run_contraption(flair, seg_path=seg, patient_id=pid, mode="compare")
        else:
            print("[INFO] Usage: python factory_pipeline.py <patient_id> or python factory_pipeline.py --test-all")
