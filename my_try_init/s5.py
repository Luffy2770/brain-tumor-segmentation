import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
      
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck (Deepest point)
        self.bottleneck = conv_block(128, 256)

        # Decoder (Upsampling + Attention Gates)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag3  = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = conv_block(256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag2  = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = conv_block(128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.ag1  = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dec1 = conv_block(64, 32)

        # Final 1x1 Conv (Outputs 1 raw logit per pixel)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Downward Pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        # Upward Pass with Attention Spotlights
        d3 = self.up3(b)
        x3 = self.ag3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        x2 = self.ag2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        x1 = self.ag1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        return self.out_conv(d1)

class PrecisionCarver:
    def __init__(self, weights_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionUNet2D(in_channels=1, out_channels=1).to(self.device)
        
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"[Station 5] Carver weights loaded: {weights_path}")
        self.model.eval()

    def carve_mask(self, tensor_4d):
        """
        Takes (1, 1, 128, 128) -> Returns (128, 128) continuous probability map (0.0 to 1.0)
        """
        with torch.inference_mode():
            raw_logits = self.model(tensor_4d)
            prob_map = torch.sigmoid(raw_logits)[0, 0]
            return prob_map