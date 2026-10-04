import numpy as np
import torch
import torch.nn.functional as F

class ResolutionRestorer:
    """
    Station 6: Upscales 128x128 probability maps back to original slice dimensions and binarizes.
    """
    def __init__(self, binary_threshold=0.5):
        self.binary_threshold = binary_threshold

    def restore_and_binarize(self, prob_map_128, target_shape):
        """
        Takes:
          - prob_map_128: PyTorch Tensor of shape (128, 128) with values [0.0 ... 1.0]
          - target_shape: tuple of original slice dimensions, e.g. (240, 240)
        Returns:
          - binary_mask: 2D NumPy array of shape (240, 240) with values 0.0 or 1.0
        """
        # Step 1: Pack back into 4D tensor (1, 1, 128, 128) so F.interpolate can process it
        tensor_4d = prob_map_128[None, None, ...]

        # Step 2: Smoothly expand up to target scan resolution (e.g. 240x240)
        upscaled_tensor = F.interpolate(
            tensor_4d,
            size=target_shape,
            mode='bilinear',
            align_corners=False
        )

        # Step 3: Unpack from 4D GPU tensor back into a flat 2D NumPy array on CPU
        upscaled_prob_2d = upscaled_tensor.squeeze().cpu().numpy()

        # Step 4: The Binary Stamping Press
        binary_mask_2d = (upscaled_prob_2d > self.binary_threshold).astype(np.float32)

        return binary_mask_2d