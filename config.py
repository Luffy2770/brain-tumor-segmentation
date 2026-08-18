import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LOCAL_BRATS_PATH = os.path.join(PROJECT_ROOT, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
SAMPLE_DATA_PATH = os.path.join(PROJECT_ROOT, "sample_data")

if os.environ.get("BRATS_DIR"):
    BRATS_DIR = os.environ.get("BRATS_DIR")
elif os.path.exists(LOCAL_BRATS_PATH):
    BRATS_DIR = LOCAL_BRATS_PATH
elif os.path.exists(SAMPLE_DATA_PATH):
    BRATS_DIR = SAMPLE_DATA_PATH
else:
    BRATS_DIR = SAMPLE_DATA_PATH

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")
OUTPUT_DIR = os.path.join(PROCESSED_DIR, "outputs")
REPORT_DIR = os.path.join(PROCESSED_DIR, "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CLASSIFIER_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_classifier.pth")
UNET_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_unet.pth")
ATTENTION_UNET_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_attention_unet.pth")
MODEL_PATH = CLASSIFIER_MODEL_PATH

MODEL_TYPE = "attention_unet"
LOSS_TYPE = "focal_tversky"
USE_DATA_AUGMENTATION = True

CHOSEN_THRESHOLD = 0.45
IMG_SIZE = 128
MIN_FRAGMENT_SIZE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXCLUDED_PATIENTS = ["BraTS20_Training_355"]
DEFAULT_PATIENT_ID = "BraTS20_Training_001"
DEFAULT_TEST_PATIENTS = [
    "BraTS20_Training_230", "BraTS20_Training_156", "BraTS20_Training_153",
    "BraTS20_Training_127", "BraTS20_Training_348", "BraTS20_Training_220"
]

CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3
CLASSIFIER_EPOCHS = 15
CLASSIFIER_PATIENCE = 3

SEG_BATCH_SIZE = 16
SEG_LR = 1e-4
SEG_EPOCHS = 25
SEG_PATIENCE = 5
