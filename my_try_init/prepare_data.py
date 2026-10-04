import os
import shutil
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from config import DATASET_DIR, DATA_DIR, PROJECT_ROOT
from s1 import ScanLoader
from s2 import SliceNormalizer

def extract_and_cache_dataset(force_extract=False):
    """
    Extracts 2D slices from 3D MRI scans and caches them in my_try_init/data/
    """
    # Check if files already exist in parent processed folder to save time if desired
    parent_processed = os.path.join(PROJECT_ROOT, "processed")
    existing_files = [
        "seg_train_images.npy", "seg_train_masks.npy", "seg_train_pids.npy",
        "seg_val_images.npy", "seg_val_masks.npy", "seg_val_pids.npy",
        "seg_test_images.npy", "seg_test_masks.npy", "seg_test_pids.npy"
    ]
    
    can_fast_sync = all(os.path.exists(os.path.join(parent_processed, f)) for f in existing_files)
    
    if can_fast_sync and not force_extract:
        print("[Prepare Data] Fast-linking existing processed dataset slices into my_try_init/data/...")
        for f in existing_files:
            src = os.path.join(parent_processed, f)
            dst = os.path.join(DATA_DIR, f)
            if not os.path.exists(dst):
                # Copy or link
                print(f" -> Caching {f}...")
                shutil.copyfile(src, dst)
        print("[Prepare Data] All dataset slices ready in my_try_init/data/!")
        return

    print(f"[Prepare Data] Scanning patient directories in: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    all_patients = sorted([
        p for p in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, p)) and p != "BraTS20_Training_355"
    ])
    
    print(f"Found {len(all_patients)} valid patient scans.")
    
    # 80% Train, 10% Validation, 10% Test
    train_p, temp_p = train_test_split(all_patients, test_size=0.2, random_state=42)
    val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

    print(f"Split: Train={len(train_p)} patients | Val={len(val_p)} patients | Test={len(test_p)} patients")

    normalizer = SliceNormalizer()

    def process_split(patient_list, split_name):
        images, masks, pids = [], [], []
        for pid in patient_list:
            pdir = os.path.join(DATASET_DIR, pid)
            flair_path = os.path.join(pdir, f"{pid}_flair.nii")
            seg_path = os.path.join(pdir, f"{pid}_seg.nii")
            
            if not (os.path.exists(flair_path) and os.path.exists(seg_path)):
                continue

            loader = ScanLoader(flair_path)
            seg_data = nib.load(seg_path).get_fdata().astype(np.float32)

            for i in range(loader.total_slices):
                mask_slice = seg_data[:, :, i]
                # Keep tumor slices for segmentation training
                if not (mask_slice > 0).any():
                    continue

                raw_slice = loader.get_slice(i)
                norm_slice = normalizer.normalize(raw_slice)
                if norm_slice is None:
                    continue

                binary_mask = (mask_slice > 0).astype(np.float32)
                images.append(norm_slice)
                masks.append(binary_mask)
                pids.append(pid)

            print(f"[{split_name}] Processed: {pid} (Total slices so far: {len(images)})")

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32), np.array(pids)

    for name, plist in [("train", train_p), ("val", val_p), ("test", test_p)]:
        imgs, msks, pids = process_split(plist, name)
        np.save(os.path.join(DATA_DIR, f"seg_{name}_images.npy"), imgs)
        np.save(os.path.join(DATA_DIR, f"seg_{name}_masks.npy"), msks)
        np.save(os.path.join(DATA_DIR, f"seg_{name}_pids.npy"), pids)
        print(f"[Done] Saved {name} set: {len(imgs)} slices to {DATA_DIR}\n")

if __name__ == "__main__":
    extract_and_cache_dataset()
