import os
import numpy as np
import nibabel as nib

class image_loader:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        self.path = path
        self.volume_data = None
        self.total_slices = 0
        
        self._load_image()
        
    def _load_image(self):
        nii_object = nib.load(self.path)
        self.volume_data = nii_object.get_fdata().astype(np.float32)
        self.total_slices = self.volume_data.shape[2]
        
        print(f"[Station 1] Loaded volume shape: {self.volume_data.shape} ({self.total_slices} slices)")
    
    def get_slice(self, slice_index):
        if slice_index < 0 or slice_index >= self.total_slices:
            raise IndexError(f"Slice index {slice_index} is out of bounds (0 to {self.total_slices - 1})")
        
        return self.volume_data[:, :, slice_index]
    
    def is_empty_slice(self, slice_2d, threshold=1e-6):
        return slice_2d.max() < threshold

# Alias for standard naming
ScanLoader = image_loader

if __name__ == "__main__":
    sample_path = "../sample_data/BraTS20_Training_001/BraTS20_Training_001_flair.nii"
    if os.path.exists(sample_path):
        loader = image_loader(sample_path)
        print(f"Test slice 75 max value: {loader.get_slice(75).max()}")
