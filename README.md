# 🧠 Cascaded Brain Tumor Segmentation System

A two-stage deep learning pipeline for automated screening and 3D volume segmentation of brain tumors from FLAIR MRI scans using **PyTorch**, **Attention U-Net**, and **Focal Tversky Loss**.

---

## 📌 Project Overview

This repository provides an end-to-end medical image analysis pipeline for MRI brain scans:
1. **Stage 1 (Slice Screening Classifier)**: A lightweight 2D CNN (`TumorClassifierCNN`) screens individual axial slices to filter out non-tumor tissue and accelerate processing.
2. **Stage 2 (Volumetric Segmentation)**: An **Attention U-Net (`AttentionUNet2D`)** with Attention Gates on skip connections predicts 2D tumor probability maps on flagged slices.
3. **Stage 3 (3D Post-Processing & Cleanup)**: SciPy 3D connected-component analysis removes small isolated noise fragments (< 50 voxels) and re-assembles the full 3D tumor volume.
4. **Diagnostics & Analytics**: Automatically generates horizontal 3D scan timeline barcodes, relative tumor presence heatmaps, per-slice CSV reports, and 3D volume mask comparison metrics (Dice, IoU, Precision, Recall).

---

## 📁 Repository Directory Structure

```text
brain-tumor-segmentation/
├── README.md                   # Project documentation & Quick Start Guide
├── config.py                   # Master configuration & automatic sample_data path resolver
├── pipeline.py                 # Primary end-to-end 2-stage cascaded pipeline runner
├── compare_predictions.py      # Volume prediction & comparison visualizer
├── compare_masks.py            # Detailed 3D mask comparison & metric evaluator
├── utils.py                    # Shared normalization, image processing & helper utilities
├── classification/             # Stage 1: Slice Screening Classifier
│   └── model.py                # TumorClassifierCNN 2D slice classifier architecture
├── segmentation/               # Stage 2: 2D Tumor Segmentation
│   ├── attention_unet.py       # AttentionUNet2D architecture with Attention Gates
│   ├── unet_model.py           # Standard UNet2D architecture
│   ├── losses.py               # Focal Tversky, Tversky, Combo & Dice loss functions
│   ├── train_seg.py            # Segmentation model training module
│   └── evaluate_seg.py         # Segmentation test set evaluation module
├── sample_data/                # Sample patient MRI scans (~10MB) for instant execution
│   ├── BraTS20_Training_001/
│   └── BraTS20_Training_230/
├── processed/                  # Pre-trained model weights & output directories
│   ├── best_attention_unet.pth # Pre-trained Attention U-Net weights
│   ├── best_classifier.pth     # Pre-trained slice classifier weights
│   └── best_unet.pth           # Pre-trained standard U-Net weights
└── .gitignore                  # Git exclusions file
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install required Python packages:

```bash
git clone https://github.com/Luffy2770/brain-tumor-segmentation.git
cd brain-tumor-segmentation
pip install torch torchvision numpy nibabel matplotlib pandas scipy
```

### 2. Run End-to-End Inference

To run full classification, segmentation, timeline barcode generation, and combined heatmap rendering on sample MRI data:

```bash
python pipeline.py
```

Outputs will be saved in `processed/outputs/` and `processed/reports/`.

### 3. Evaluate Ground-Truth Mask Comparison

To compute 3D Volume Dice, IoU, Precision, Recall, and 4-panel error discrepancy maps against ground-truth masks:

```bash
python compare_masks.py
```

---

## 📊 Benchmark Results

| Model Architecture | Loss Function | 3D Volume Dice | Peak Volume Dice | Recall (Sensitivity) |
| :--- | :--- | :--- | :--- | :--- |
| **Attention U-Net** | Focal Tversky Loss | **89.7%** | **94.6%** | **99.6%** |
| Standard U-Net | Soft Dice Loss | 88.3% | 93.5% | 94.3% |

- **Dataset**: MICCAI BraTS 2020 Training Data (369 Patient Scans).
- **Hardware Acceleration**: NVIDIA RTX GPU supported (`cuda:0`).
