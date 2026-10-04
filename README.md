# 🧠 Automated 8-Station Brain Tumor Segmentation Contraption

A cascaded, leak-free deep learning pipeline for automated screening and 3D volume segmentation of brain tumors from axial FLAIR MRI scans using **PyTorch**, **Attention U-Net**, and **Focal Tversky Loss**.

---

## 🏗️ System Overview: The 8-Station Pipeline

Instead of feeding massive 3D MRI volumes into a heavy 3D neural network, our system operates like an automated factory assembly line, screening out non-tumor slices with a lightweight 2D classifier before routing candidate slices to an Attention U-Net:

```text
[Raw 3D MRI Scan] (.nii)
       │
       ▼
[Station 1: ScanLoader (s1.py)] ──── Loads 3D volume, extracts 2D axial slices
       │
       ▼
[Station 2: SliceNormalizer (s2.py)] ── Z-score normalization on non-zero brain pixels
       │
       ▼
[Station 3: TensorFunnel (s3.py)] ───── Interpolates slice to 128x128 4D GPU Tensor
       │
       ▼
[Station 4: SorterGate (s4.py)] ─────── Lightweight 2D CNN filters out healthy slices
       ├───────────────────────────────┐
       ▼ [Suspicious: Red Belt]        ▼ [Healthy: Green Belt]
[Station 5: PrecisionCarver (s5.py)]   [Bypasses heavy machinery!]
(Attention U-Net highlights tumor)     │
       │                               │
       ▼                               │
[Station 6: ResolutionRestorer (s6.py)]│
(Bilinear upscaling back to 240x240)   │
       │                               │
       └───────────────┬───────────────┘
                       ▼
[Station 7: VolumetricSieve (s7.py)] ── 3D Connected Components sweeps away dust (< 50 voxels)
                       │
               ┌───────┴───────┐
               ▼               ▼
      [Skull Air Mask]  [Station 8: DiagnosticPackager (s8.py)]
      (Zeroes air)     (Exports comparison PNG, slice bar graph, CSV)
```

---

## 📁 Repository Directory Structure

```text
brain-tumor-segmentation/
├── my_try_init/                    # Core 8-station production pipeline
│   ├── config.py                   # Master configuration & hyperparameters
│   ├── s1.py ... s8.py             # Modular stations S1 to S8
│   ├── factory_pipeline.py         # Primary end-to-end pipeline runner
│   ├── losses.py                   # Focal Tversky & Dice loss definitions
│   ├── check_leakage.py            # Patient-level zero-leakage audit script
│   ├── generate_analytics.py       # Confusion matrices & benchmark visualizer
│   ├── train_classifier.py         # Stage 4 2D CNN training routine
│   ├── train_carver.py             # Stage 5 Attention U-Net training routine
│   ├── PROJECT_FACTS.md            # Technical specifications & facts sheet
│   ├── README.md                   # Dedicated package documentation
│   ├── models/                     # Pre-trained model weights (< 10 MB each)
│   │   ├── best_classifier.pth     # Stage 4 trained weights
│   │   └── best_attention_unet.pth # Stage 5 trained weights
│   └── outputs/                    # Benchmark evaluations & sample results
├── sample_data/                    # Sample MRI scans for instant execution
│   ├── BraTS20_Training_001/       # Complete multi-modal scan & ground truth
│   └── BraTS20_Training_230/       # Complete multi-modal scan & ground truth
├── Presentation_Brain_Tumor_Contraption.ipynb # Interactive step-by-step walkthrough
├── requirements.txt                # Python package dependencies
└── README.md                       # Main repository guide
```

---

## 🚀 Quick Start Guide for Teammates & Friends

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Luffy2770/brain-tumor-segmentation.git
cd brain-tumor-segmentation
pip install -r requirements.txt
```

### 2. Run Pipeline on Included Sample Data (Out-of-the-Box)

The repository comes with pre-packaged sample patient volumes in `sample_data/` and pre-trained weights in `my_try_init/models/`. You can run inference immediately without downloading massive datasets:

```bash
# Compare mode on sample patient 001 (evaluates against ground-truth mask)
python my_try_init/factory_pipeline.py BraTS20_Training_001 --mode compare

# Compare mode on sample patient 230
python my_try_init/factory_pipeline.py BraTS20_Training_230 --mode compare
```

Or run directly from inside `my_try_init`:
```bash
cd my_try_init
python factory_pipeline.py BraTS20_Training_001 --mode compare
```

### 3. Inspect the Generated Outputs

Results are automatically saved to `my_try_init/outputs/<patient_id>/`:
- `comparison.png`: 3-panel visualization showing Normal Scan, Ground-Truth Mask, and Predicted Mask (with tumor volume in mL and voxel counts).
- `slice_distribution.png`: Per-slice bar chart comparing ground-truth tumor voxels vs predicted voxels across the entire 155-slice axial stack.
- `slice_report.csv`: Detailed slice-by-slice tabular audit with classification probabilities and segmentation voxel counts.

### 4. Interactive Presentation Notebook

Open `Presentation_Brain_Tumor_Contraption.ipynb` in VS Code or Jupyter Lab to step through each of the 8 stations interactively with inline visualizations.

---

## 📊 Empirical Holdout Benchmark (37 Test Patients)

The system was evaluated on a strictly holdout test set of 37 BraTS2020 patients never seen during training:

| Metric | Measured Value |
| :--- | :--- |
| **Mean 3D Volume Dice** | **82.02%** |
| **Median 3D Volume Dice** | **86.36%** |
| **Voxel Recall (Sensitivity)** | **88.89%** |
| **Voxel Precision (PPV)** | **78.01%** |
| **3D Volume IoU** | **71.74%** |
| **Stage 4 Classifier Sensitivity** | **89.52%** |
| **Stage 4 Classifier Specificity** | **94.01%** |
| **Zero Patient-Level Leakage** | **0 patient overlap (Verified)** |

> For complete mathematical definitions, layer parameters, and slice audits, see [`my_try_init/PROJECT_FACTS.md`](my_try_init/PROJECT_FACTS.md).
