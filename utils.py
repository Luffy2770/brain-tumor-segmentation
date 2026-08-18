import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage


def normalize_slice(raw_slice):
    brain_pixels = raw_slice[raw_slice > 0]
    if brain_pixels.size == 0:
        return None
    mean, std = brain_pixels.mean(), brain_pixels.std()
    return (raw_slice - mean) / (std + 1e-8)


def clean_small_fragments(mask_stack, min_size=50):
    labeled, num_features = ndimage.label(mask_stack > 0)
    if num_features == 0:
        return mask_stack
    sizes = ndimage.sum(mask_stack > 0, labeled, range(1, num_features + 1))
    cleaned = np.zeros_like(mask_stack)
    for region_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            cleaned[labeled == region_id] = mask_stack[labeled == region_id]
    return cleaned


def to_tensor_batch(norm_slice, img_size=128, device="cpu"):
    img = torch.from_numpy(norm_slice)[None, None, ...].float().to(device)
    return F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)


def get_tumor_span_stats(slice_probs, threshold=0.35):
    slice_probs = np.array(slice_probs)
    predictions = (slice_probs > threshold).astype(int)
    total_slices = len(slice_probs)
    flagged_indices = np.where(predictions == 1)[0]
    flagged_count = len(flagged_indices)

    if flagged_count > 0:
        start_slice = int(flagged_indices.min())
        end_slice = int(flagged_indices.max())
        peak_slice = int(np.argmax(slice_probs))
        peak_prob = float(slice_probs[peak_slice])
        height_pct = round((flagged_count / total_slices) * 100, 1)
    else:
        start_slice, end_slice, peak_slice, peak_prob, height_pct = None, None, None, 0.0, 0.0

    return {
        "total_slices": total_slices,
        "flagged_count": flagged_count,
        "start_slice": start_slice,
        "end_slice": end_slice,
        "peak_slice": peak_slice,
        "peak_prob": round(peak_prob, 4),
        "height_pct": height_pct
    }


def generate_scan_timeline(slice_probs, threshold=0.35, out_path=None):
    slice_probs = np.array(slice_probs)
    total_slices = len(slice_probs)
    
    status = np.zeros(total_slices)
    for i, p in enumerate(slice_probs):
        if p > threshold:
            status[i] = 2.0
        elif p > 0.001:
            status[i] = 1.0

    fig, ax = plt.subplots(figsize=(10, 1.8))
    cmap = plt.matplotlib.colors.ListedColormap(['#dcdcdc', '#2ecc71', '#e74c3c'])
    ax.imshow(status[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=2)

    ax.set_yticks([])
    ax.set_xlabel("Slice Index (Inferior -> Superior)", fontsize=10, fontweight='bold')
    ax.set_title("3D Scan Tumor Timeline Barcode", fontsize=11, fontweight='bold')
    
    if (status == 2).any():
        peak_idx = int(np.argmax(slice_probs))
        ax.axvline(peak_idx, color='black', linestyle='--', linewidth=1.5)
        ax.text(peak_idx, -0.6, f"Peak #{peak_idx}", color='black', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def select_smart_peak_slices(scan, slice_probs, predictions, num_samples=8):
    total_slices = scan.shape[2]
    tumor_indices = np.where(predictions == 1)[0]
    
    if len(tumor_indices) > 0:
        sorted_by_prob = tumor_indices[np.argsort(-slice_probs[tumor_indices])]
        peak_indices = sorted_by_prob[:min(num_samples, len(sorted_by_prob))]
        if len(peak_indices) < num_samples:
            context_indices = np.linspace(0, total_slices - 1, num_samples - len(peak_indices)).astype(int)
            selected = np.unique(np.concatenate([peak_indices, context_indices]))
        else:
            selected = peak_indices
        selected.sort()
        return selected[:num_samples]
    else:
        return np.linspace(0, total_slices - 1, num_samples).astype(int)


def load_segmenter_model(device="cpu"):
    from config import MODEL_TYPE, UNET_MODEL_PATH, ATTENTION_UNET_MODEL_PATH
    from unet_model import UNet2D
    from attention_unet import AttentionUNet2D

    if MODEL_TYPE == "attention_unet" and os.path.exists(ATTENTION_UNET_MODEL_PATH):
        model = AttentionUNet2D(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(torch.load(ATTENTION_UNET_MODEL_PATH, map_location=device))
    elif os.path.exists(UNET_MODEL_PATH):
        model = UNet2D(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
    else:
        model = AttentionUNet2D(in_channels=1, out_channels=1).to(device)

    model.eval()
    return model
