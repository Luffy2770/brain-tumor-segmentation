import numpy as np

class SliceNormalizer:
    """
    Station 2: Washes individual 2D slices to standardized brightness and contrast.
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def normalize(self, raw_slice, mask_background=False):
        """
        Takes a raw 2D numpy slice and returns a standardized (Z-score normalized) slice.
        """
        brain_pixels = raw_slice[raw_slice > 0]
        
        if brain_pixels.size == 0:
            return None
        
        mean = brain_pixels.mean()
        std = brain_pixels.std()
        
        normalized_slice = (raw_slice - mean) / (std + self.epsilon)
        
        if mask_background:
            normalized_slice[raw_slice == 0] = 0.0
            
        return normalized_slice.astype(np.float32)
