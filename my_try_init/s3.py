import numpy as np
import torch
import torch.nn.functional as F

class TensorFunnel:
    """
    Station 3: Packages 2D image slices into standard 4D PyTorch tensors and resizes them.
    """
    def __init__(self, target_size=128, device=None):
        self.target_size = target_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"[Station 3] Funnel configured for target size: {self.target_size}x{self.target_size} on {self.device}")

    def prepare_slice(self, clean_slice):
        """
        Converts a 2D clean numpy slice (H, W) -> 4D PyTorch tensor (1, 1, target_size, target_size)
        """
        tensor_4d = torch.from_numpy(clean_slice)[None, None, ...]
        tensor_4d = tensor_4d.float().to(self.device)
        resized_tensor = F.interpolate(
            tensor_4d, 
            size=(self.target_size, self.target_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        return resized_tensor