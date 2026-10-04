import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        Tversky Loss (Salehi et al., 2017).
        alpha=0.3 penalizes False Positives (over-segmentation).
        beta=0.7 penalizes False Negatives (missed tumors - critical for medical safety).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        """
        Focal Tversky Loss (Abraham & Khan, 2019: 'A Novel Focal Tversky Loss Function with Improved Attention U-Net').
        Formula: FTL = (1 - Tversky_Index) ** (1 / gamma)
        With gamma = 4/3 (~1.33), the exponent 1/gamma = 0.75 non-linearly focuses on hard, ambiguous boundary voxels.

        NOTE ON REPRODUCIBILITY / WEIGHTS:
        The current checkpoint (best_attention_unet.pth, saved 10/3/2026 4:03 PM) was trained with
        Tversky beta=0.7 and focal exponent 1.33. The formula was subsequently corrected to the
        paper's exact 1/gamma = 0.75 formulation without retraining the existing checkpoint.
        """
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits, targets):
        tversky_loss = self.tversky(logits, targets)
        # Exponent is 1 / gamma (0.75 for gamma=1.33)
        return torch.pow(tversky_loss, 1.0 / self.gamma)
