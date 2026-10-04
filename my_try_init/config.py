import os
import torch

# Base folder for this self-contained package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Dataset Paths
DATASET_DIR = os.path.join(PROJECT_ROOT, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")

# Folders inside my_try_init
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Auto-create all necessary folders
for folder in [MODELS_DIR, DATA_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Model Checkpoint Paths
CLASSIFIER_WEIGHTS = os.path.join(MODELS_DIR, "best_classifier.pth")
CARVER_WEIGHTS = os.path.join(MODELS_DIR, "best_attention_unet.pth")

# Hardware Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IMG_SIZE = 128
CLASSIFIER_THRESHOLD = 0.45
SEG_BINARY_THRESHOLD = 0.5
MIN_FRAGMENT_SIZE = 50

# Classifier Training Settings
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3
CLASSIFIER_EPOCHS = 15
CLASSIFIER_PATIENCE = 4

# Carver (Attention U-Net) Training Settings
SEG_BATCH_SIZE = 16
SEG_LR = 1e-4
SEG_EPOCHS = 25
SEG_PATIENCE = 5
LOSS_TYPE = "focal_tversky"  # options: 'focal_tversky', 'tversky', 'combo', 'dice'
USE_AUGMENTATION = True
