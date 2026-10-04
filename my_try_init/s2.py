import numpy as np

class SliceNormalizer:
    """
    Station 2: Washes individual 2D slices to standardized brightness and contrast.
    Safely rejects slices with insufficient tissue (< min_pixels) or flat variance (< min_std).
    """
    def __init__(self, epsilon=1e-8, min_pixels=10, min_std=1e-4):
        self.epsilon = epsilon
        self.min_pixels = min_pixels
        self.min_std = min_std

    def normalize(self, raw_slice, mask_background=False):
        """
        Takes a raw 2D numpy slice and returns a standardized (Z-score normalized) slice.
        Returns None if slice contains fewer than min_pixels non-zero voxels or std < min_std.
        """
        brain_pixels = raw_slice[raw_slice > 0]
        
        if brain_pixels.size < self.min_pixels:
            return None
        
        mean = brain_pixels.mean()
        std = brain_pixels.std()
        
        if std < self.min_std:
            return None
        
        normalized_slice = (raw_slice - mean) / (std + self.epsilon)
        
        if mask_background:
            normalized_slice[raw_slice == 0] = 0.0
            
        return normalized_slice.astype(np.float32)
