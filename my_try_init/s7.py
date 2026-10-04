import numpy as np
from scipy import ndimage

class VolumetricSieve:
    """
    Station 7: Cleans small, disconnected noise fragments from the 3D mask stack.
    """
    def __init__(self, min_size=50):
        # Any 3D island with fewer voxels than min_size gets erased
        self.min_size = min_size
        print(f"[Station 7] Volumetric Sieve online (Dust threshold: {self.min_size} voxels)")

    def clean_volume(self, mask_3d_stack):
        """
        Takes:
          - mask_3d_stack: 3D NumPy array of shape (H, W, Depth) containing 0s and 1s
        Returns:
          - cleaned_stack: 3D NumPy array with tiny floating fragments erased
        """
        # Step 1: Detect all independent 3D islands (connected components)
        # labeled: array where every island has a unique number (1, 2, 3...)
        # num_features: total number of islands found
        labeled, num_features = ndimage.label(mask_3d_stack > 0)

        # Safety Check: If the entire brain was completely healthy (0 tumors found)
        if num_features == 0:
            return mask_3d_stack

        # Step 2: Measure the voxel count of every single island
        # range(1, num_features + 1) checks islands 1 through N
        island_sizes = ndimage.sum(mask_3d_stack > 0, labeled, range(1, num_features + 1))

        # Step 3: Prepare a clean empty 3D crate
        cleaned_stack = np.zeros_like(mask_3d_stack)

        # Step 4: Keep only islands that meet our minimum size threshold
        removed_count = 0
        for region_id, size in enumerate(island_sizes, start=1):
            if size >= self.min_size:
                # Solid tumor mass: Transfer to the clean crate
                cleaned_stack[labeled == region_id] = mask_3d_stack[labeled == region_id]
            else:
                # Tiny dust speckle: Ignored (left as 0)
                removed_count += 1

        if removed_count > 0:
            print(f"[Station 7] Swept away {removed_count} floating noise fragments (< {self.min_size} voxels).")

        return cleaned_stack