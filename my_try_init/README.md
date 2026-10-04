# 🏭 8-Station Brain Tumor Segmentation Pipeline (`my_try_init`)

An end-to-end, leak-free, two-stage deep learning pipeline for automated screening and 3D volume segmentation of brain tumors from axial FLAIR MRI scans using **PyTorch**, **Attention U-Net**, and **Focal Tversky Loss**.

---

## 🏗️ The 8-Station Assembly Line Architecture

Instead of feeding massive 3D MRI volumes into a heavy 3D network, the pipeline processes scans through a sequence of 8 specialized, modular stations:

```text
[Raw 3D MRI Scan] (.nii)
       │
       ▼
[Station 1: ScanLoader (s1.py)] ──── Loads 3D volume into memory, extracts 2D axial slices
       │
       ▼
[Station 2: SliceNormalizer (s2.py)] ── Nonzero brain-pixel Z-score normalization
       │
       ▼
[Station 3: TensorFunnel (s3.py)] ───── Interpolates slice to 128x128 4D GPU Tensor
       │
       ▼
[Station 4: SorterGate (s4.py)] ─────── Lightweight 2D CNN classifier filters healthy slices
       ├───────────────────────────────┐
       ▼ [Suspicious: Red Belt]        ▼ [Healthy: Green Belt]
[Station 5: PrecisionCarver (s5.py)]   [Bypasses heavy neural segmentation]
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

## 📁 Station Directory & Script Breakdown

| Station / Script | Role / Description |
| :--- | :--- |
| [`s1.py`](file:///s1.py) | **Station 1 (Ingestion)**: Loads 3D NIfTI scans (`nibabel`), exposes slices via `get_slice(idx)`, and filters all-zero air slices via `is_empty_slice()`. |
| [`s2.py`](file:///s2.py) | **Station 2 (Preprocessing)**: Z-score normalization computed strictly over non-zero brain pixels. |
| [`s3.py`](file:///s3.py) | **Station 3 (Funnel)**: Converts 2D NumPy slices to 4D PyTorch tensors `(1, 1, 128, 128)` with bilinear interpolation. |
| [`s4.py`](file:///s4.py) | **Station 4 (Sorter Gate)**: 3-layer CNN classifier (64k params) flagging slices above threshold `p > 0.45`. |
| [`s5.py`](file:///s5.py) | **Station 5 (Carver)**: 4-depth Attention U-Net (653k params) with attention gates producing tumor probability maps. |
| [`s6.py`](file:///s6.py) | **Station 6 (Restorer)**: Upscales 128x128 probability maps back to original 240x240 resolution and binarizes at threshold `p > 0.5`. |
| [`s7.py`](file:///s7.py) | **Station 7 (Sieve)**: SciPy 3D connected-component labeling eliminating disconnected floating artifacts `< 50` voxels. |
| [`s8.py`](file:///s8.py) | **Station 8 (Packager)**: Produces multi-panel diagnostic PNGs, per-slice tumor bar graphs, and slice audit CSV reports. |
| [`factory_pipeline.py`](file:///factory_pipeline.py) | **Master Pipeline**: Single shared execution engine for single-scan inference, evaluation, and batch benchmarking. |
| [`config.py`](file:///config.py) | Centralized hyperparameter configuration, path management, and hardware selection. |
| [`losses.py`](file:///losses.py) | Loss functions: Focal Tversky Loss, Tversky Loss, Dice Loss. |
| [`check_leakage.py`](file:///check_leakage.py) | Zero-leakage audit script verifying disjoint patient splits between training, validation, and testing. |
| [`generate_analytics.py`](file:///generate_analytics.py) | Generates confusion matrices, benchmark distribution plots, and performance analytics. |
| [`PROJECT_FACTS.md`](file:///PROJECT_FACTS.md) | Exhaustive empirical project facts and technical reference document. |

---

## ⚡ Quick Start & Usage

Both models are pre-trained and saved in `models/`:
- `models/best_classifier.pth` (~8.5 MB)
- `models/best_attention_unet.pth` (~7.9 MB)

### 1. Compare Mode (Scan + Ground Truth Mask)
Evaluates predictions against ground truth and outputs 3D Volume Dice, IoU, sensitivity, precision, comparison PNG, and per-slice distribution bar graph:

```bash
# Using patient ID (auto-resolved from sample_data/ or main dataset)
python factory_pipeline.py BraTS20_Training_001 --mode compare

# Or passing direct directory path
python factory_pipeline.py ../sample_data/BraTS20_Training_230 --mode compare
```

### 2. Predict Mode (Scan Only, No Ground Truth Needed)
Performs inference on an unannotated patient scan:

```bash
python factory_pipeline.py BraTS20_Training_001 --mode predict
```

### 3. Verify Patient Leakage
```bash
python check_leakage.py
```

---

## 📊 Benchmark Summary (37-Patient Holdout Test Set)

Evaluated on 37 strictly holdout test patients (unseen during training of both the Classifier and Attention U-Net):

| Metric | Measured Value |
| :--- | :--- |
| **Mean 3D Volume Dice** | **82.02%** |
| **Median 3D Volume Dice** | **86.36%** |
| **Mean Voxel Recall (Sensitivity)** | **88.89%** |
| **Mean Voxel Precision (PPV)** | **78.01%** |
| **Mean 3D Volume IoU** | **71.74%** |
| **Stage 4 Classifier Sensitivity** | **89.52%** (2,273 / 2,539 tumor slices) |
| **Stage 4 Classifier Specificity** | **94.01%** (2,414 / 2,568 healthy slices) |
| **Zero Patient-Level Leakage** | **Verified (0 overlap)** |

---

## 🔬 Hardware & Compatibility
- Supports NVIDIA CUDA GPUs and CPU fallback.
- Python 3.9+ / 3.10 / 3.11 with PyTorch 2.0+.
