import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class TumorClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class SorterGate:
    """
    Station 4: Inspects packaged tensors and routes them to Red Belt (Tumor) or Green Belt (Healthy).
    """
    def __init__(self, weights_path=None, threshold=0.45, device=None):
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = TumorClassifierCNN().to(self.device)
        
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"[Station 4] Loaded trained weights from: {weights_path}")
        else:
            print("[Station 4] Running with uninitialized weights (Train first to use accurate weights).")
            
        self.model.eval()

    def inspect_slice(self, tensor_4d):
        """
        Takes a (1, 1, 128, 128) tensor and returns (tumor_probability, should_divert_to_carver)
        """
        with torch.inference_mode():
            logits = self.model(tensor_4d)
            probabilities = F.softmax(logits, dim=1)
            tumor_prob = probabilities[0, 1].item()
            is_suspicious = tumor_prob > self.threshold
            return tumor_prob, is_suspicious
