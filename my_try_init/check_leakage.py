import os
import numpy as np

def verify_patient_splits():
    """
    Formally audits patient-level partition across Stage 4 (Classifier) and
    Stage 5 (Attention U-Net Carver) against the 37 test patients.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    seg_train_pids = set(np.unique(np.load(os.path.join(data_dir, "seg_train_pids.npy"))))
    seg_val_pids   = set(np.unique(np.load(os.path.join(data_dir, "seg_val_pids.npy"))))
    test_pids      = set(np.unique(np.load(os.path.join(data_dir, "seg_test_pids.npy"))))

    cls_train_pids = set(np.unique(np.load(os.path.join(data_dir, "cls_train_pids.npy")))) if os.path.exists(os.path.join(data_dir, "cls_train_pids.npy")) else set()
    cls_val_pids   = set(np.unique(np.load(os.path.join(data_dir, "cls_val_pids.npy")))) if os.path.exists(os.path.join(data_dir, "cls_val_pids.npy")) else set()

    print("============================================================")
    print("      DATA LEAKAGE AUDIT: PATIENT-LEVEL SPLIT VERIFICATION")
    print("============================================================")
    print(f" Total Unique Patients Evaluated  : {len(seg_train_pids) + len(seg_val_pids) + len(test_pids)}")
    print(f" Carver Train Patients            : {len(seg_train_pids)}")
    print(f" Carver Val Patients              : {len(seg_val_pids)}")
    if cls_train_pids:
        print(f" Classifier Train Patients        : {len(cls_train_pids)}")
        print(f" Classifier Val Patients          : {len(cls_val_pids)}")
    print(f" Test Patients (Evaluation Set)   : {len(test_pids)}")
    print("------------------------------------------------------------")

    # Audit Carver Splits
    carver_train_leak = seg_train_pids.intersection(test_pids)
    carver_val_leak   = seg_val_pids.intersection(test_pids)

    print(f" [Carver] Train & Test Overlap   : {len(carver_train_leak)} patients")
    print(f" [Carver] Val & Test Overlap     : {len(carver_val_leak)} patients")

    # Audit Classifier Splits
    cls_train_leak = cls_train_pids.intersection(test_pids)
    cls_val_leak   = cls_val_pids.intersection(test_pids)

    if cls_train_pids:
        print(f" [Classifier] Train & Test Overlap: {len(cls_train_leak)} patients")
        print(f" [Classifier] Val & Test Overlap  : {len(cls_val_leak)} patients")
    print("------------------------------------------------------------")

    all_clean = (
        len(carver_train_leak) == 0 and
        len(carver_val_leak) == 0 and
        len(cls_train_leak) == 0 and
        len(cls_val_leak) == 0
    )

    if all_clean:
        print("[AUDIT PASSED] ZERO PATIENT LEAKAGE DETECTED.")
        print(f"None of the {len(test_pids)} test patients appear in the training or validation")
        print("data for either the Stage 4 Classifier or the Stage 5 Attention U-Net.")
    else:
        print("[AUDIT FAILED] CRITICAL LEAKAGE DETECTED!")
    print("============================================================\n")

if __name__ == "__main__":
    verify_patient_splits()
