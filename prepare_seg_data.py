import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from config import BRATS_DIR, PROCESSED_DIR, EXCLUDED_PATIENTS

patient_ids = sorted(os.listdir(BRATS_DIR))
patient_ids = [p for p in patient_ids if p not in EXCLUDED_PATIENTS]  # match classifier's set
unique_patients = np.array(patient_ids)

# SAME split as classifier -- same random_state=42, so test patients match exactly
train_p, temp_p = train_test_split(unique_patients, test_size=0.2, random_state=42)
val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

def extract_seg_slices(patient_list, split_name):
    images, masks, pids = [], [], []
    for pid in patient_list:
        patient_dir = os.path.join(BRATS_DIR, pid)
        flair_path = os.path.join(patient_dir, f"{pid}_flair.nii")
        seg_path = os.path.join(patient_dir, f"{pid}_seg.nii")
        if not (os.path.exists(flair_path) and os.path.exists(seg_path)):
            continue

        scan = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        for i in range(scan.shape[2]):
            scan_slice = scan[:, :, i]
            mask_slice = seg[:, :, i]

            if not (mask_slice > 0).any():   # only keep tumor-present slices
                continue

            brain_pixels = scan_slice[scan_slice > 0]
            if brain_pixels.size == 0:
                continue
            mean, std = brain_pixels.mean(), brain_pixels.std()
            norm_slice = (scan_slice - mean) / (std + 1e-8)

            binary_mask = (mask_slice > 0).astype(np.float32)  # 1=tumor, 0=background

            images.append(norm_slice.astype(np.float32))
            masks.append(binary_mask)
            pids.append(pid)

        print(f"[{split_name}] done: {pid} -- running total: {len(images)}")

    return np.array(images), np.array(masks), np.array(pids)

if __name__ == "__main__":
    for name, plist in [("train", train_p), ("val", val_p), ("test", test_p)]:
        imgs, msks, pids = extract_seg_slices(plist, name)
        np.save(os.path.join(PROCESSED_DIR, f"seg_{name}_images.npy"), imgs)
        np.save(os.path.join(PROCESSED_DIR, f"seg_{name}_masks.npy"), msks)
        np.save(os.path.join(PROCESSED_DIR, f"seg_{name}_pids.npy"), pids)
        print(f"Saved {name}: {len(imgs)} tumor-slices\n")