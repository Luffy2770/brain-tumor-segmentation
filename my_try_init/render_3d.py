"""
================================================================================
3D VOLUMETRIC RENDERING STATION: INTERACTIVE BRAIN & TUMOR VISUALIZATION
================================================================================
A standalone module that extracts high-resolution 3D polygonal surface meshes
from MRI scans and segmentations (using Marching Cubes), and renders them in:
  1. An interactive, standalone 3D WebGL HTML application (via Plotly)
     with 360-degree rotation, zoom, glass-brain transparency, and camera presets.
  2. A 4-panel static 3D snapshot PNG (Axial, Coronal, Sagittal, Isometric)
     suitable for inclusion in clinical reports and presentations.

Usage:
  python my_try_init/render_3d.py [PATIENT_ID] [--open] [--skip-brain]
  python render_3d.py BraTS20_Training_001 --open
================================================================================
"""

import os
import sys
import argparse
import webbrowser
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage import measure
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Ensure my_try_init directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    DATASET_DIR, PROJECT_ROOT, OUTPUTS_DIR, REPORTS_DIR, DEVICE,
    CLASSIFIER_WEIGHTS, CARVER_WEIGHTS, CLASSIFIER_THRESHOLD,
    SEG_BINARY_THRESHOLD, MIN_FRAGMENT_SIZE
)
from factory_pipeline import find_patient_files, run_contraption


def extract_surface_mesh(binary_volume, level=0.5, step_size=1, smooth_sigma=0.0):
    """
    Extracts a 3D surface mesh using Marching Cubes.
    Optionally applies Gaussian smoothing to create an organic, rounded surface.
    """
    vol = binary_volume.astype(np.float32)
    if smooth_sigma > 0:
        vol = ndi.gaussian_filter(vol, sigma=smooth_sigma)
    
    # Marching cubes isosurface extraction
    verts, faces, normals, _ = measure.marching_cubes(
        vol, level=level, step_size=step_size, allow_degenerate=False
    )
    return verts, faces, normals


def build_glass_brain_mesh(flair_data, step_size=2, intensity_threshold=15.0):
    """
    Constructs a smooth, organic translucent 'glass brain' outer shell.
    """
    # 1. Non-zero brain foreground mask
    brain_mask = (flair_data > intensity_threshold).astype(np.float32)
    
    # 2. Morphological opening to clean outer air/skull noise
    struct = ndi.generate_binary_structure(3, 1)
    brain_mask = ndi.binary_opening(brain_mask, structure=struct)
    
    # 3. Gaussian smoothing to make the brain boundary smooth and organic
    brain_smooth = ndi.gaussian_filter(brain_mask.astype(np.float32), sigma=1.2)
    
    # 4. Marching cubes
    verts, faces, normals = extract_surface_mesh(
        brain_smooth, level=0.35, step_size=step_size, smooth_sigma=0.0
    )
    return verts, faces, normals


def create_interactive_3d_html(
    patient_id,
    metrics,
    verts_pred=None, faces_pred=None,
    verts_gt=None, faces_gt=None,
    verts_brain=None, faces_brain=None,
    output_html_path="render_3d.html"
):
    """
    Builds a standalone, interactive 3D WebGL visualization with Plotly.
    Includes glass-brain opacity, camera view presets, and scorecard HUD.
    """
    fig = go.Figure()

    # 1. Translucent Glass Brain Shell
    if verts_brain is not None and faces_brain is not None:
        fig.add_trace(go.Mesh3d(
            x=verts_brain[:, 0],
            y=verts_brain[:, 1],
            z=verts_brain[:, 2],
            i=faces_brain[:, 0],
            j=faces_brain[:, 1],
            k=faces_brain[:, 2],
            color="#64748b",  # Sleek slate gray
            opacity=0.12,
            name="Translucent Brain Anatomy",
            hoverinfo="name",
            lighting=dict(
                ambient=0.6,
                diffuse=0.5,
                specular=0.1,
                roughness=0.6,
                fresnel=0.3
            ),
            showlegend=True
        ))

    # 2. Ground Truth Tumor Mask (if available)
    if verts_gt is not None and faces_gt is not None:
        gt_vol_str = f"{metrics.get('true_ml', 0):.1f} mL" if metrics else ""
        fig.add_trace(go.Mesh3d(
            x=verts_gt[:, 0],
            y=verts_gt[:, 1],
            z=verts_gt[:, 2],
            i=faces_gt[:, 0],
            j=faces_gt[:, 1],
            k=faces_gt[:, 2],
            color="#00f5d4",  # High-visibility neon cyan/teal
            opacity=0.60,
            name=f"Ground Truth Tumor ({gt_vol_str})",
            hoverinfo="name",
            lighting=dict(
                ambient=0.4,
                diffuse=0.7,
                specular=0.8,
                roughness=0.2
            ),
            showlegend=True
        ))

    # 3. Predicted Tumor Mask (Station 5 Carver + Station 7 Sieve)
    if verts_pred is not None and faces_pred is not None:
        pred_vol_str = f"{metrics.get('pred_ml', 0):.1f} mL" if metrics else ""
        dice_str = f"Dice: {metrics.get('dice', 0)*100:.1f}%" if metrics and 'dice' in metrics else ""
        label = f"Predicted Tumor ({pred_vol_str} | {dice_str})".strip()
        fig.add_trace(go.Mesh3d(
            x=verts_pred[:, 0],
            y=verts_pred[:, 1],
            z=verts_pred[:, 2],
            i=faces_pred[:, 0],
            j=faces_pred[:, 1],
            k=faces_pred[:, 2],
            color="#ff0055",  # Vibrant surgical magenta/crimson
            opacity=0.88,
            name=label,
            hoverinfo="name",
            lighting=dict(
                ambient=0.45,
                diffuse=0.75,
                specular=0.95,
                roughness=0.25
            ),
            showlegend=True
        ))

    # Title & Subtitle Banner
    dice_val = f"{metrics.get('dice', 0)*100:.2f}%" if metrics and 'dice' in metrics else "N/A"
    iou_val = f"{metrics.get('iou', 0)*100:.2f}%" if metrics and 'iou' in metrics else "N/A"
    sens_val = f"{metrics.get('recall', 0)*100:.2f}%" if metrics and 'recall' in metrics else "N/A"
    prec_val = f"{metrics.get('precision', 0)*100:.2f}%" if metrics and 'precision' in metrics else "N/A"
    
    title_text = (
        f"<b>3D Volumetric Brain Tumor Reconstruction</b> — <span style='color:#38bdf8;'>{patient_id}</span><br>"
        f"<span style='font-size:13px; color:#94a3b8; font-weight:normal;'>"
        f"3D Volume Dice: <b style='color:#f8fafc;'>{dice_val}</b> | "
        f"IoU: <b style='color:#f8fafc;'>{iou_val}</b> | "
        f"Sensitivity: <b style='color:#f8fafc;'>{sens_val}</b> | "
        f"Precision: <b style='color:#f8fafc;'>{prec_val}</b>"
        f"</span>"
    )

    # Interactive Camera Preset Buttons
    camera_presets = [
        dict(
            label="🎲 3D Perspective",
            method="relayout",
            args=["scene.camera", dict(
                eye=dict(x=1.65, y=1.65, z=1.25),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )]
        ),
        dict(
            label="🧠 Superior (Axial / Top)",
            method="relayout",
            args=["scene.camera", dict(
                eye=dict(x=0.0, y=0.0, z=2.4),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0)
            )]
        ),
        dict(
            label="👂 Lateral (Sagittal / Side)",
            method="relayout",
            args=["scene.camera", dict(
                eye=dict(x=2.4, y=0.0, z=0.0),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )]
        ),
        dict(
            label="👁️ Anterior (Coronal / Front)",
            method="relayout",
            args=["scene.camera", dict(
                eye=dict(x=0.0, y=2.4, z=0.0),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )]
        )
    ]

    # Layout Aesthetics
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(color="#f8fafc", family="Inter, system-ui, -apple-system, sans-serif", size=18),
            x=0.04,
            y=0.96
        ),
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        margin=dict(l=0, r=0, b=0, t=75),
        scene=dict(
            xaxis=dict(
                title="Sagittal (X)",
                backgroundcolor="#090d16",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
                color="#94a3b8"
            ),
            yaxis=dict(
                title="Coronal (Y)",
                backgroundcolor="#090d16",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
                color="#94a3b8"
            ),
            zaxis=dict(
                title="Axial Depth (Z)",
                backgroundcolor="#090d16",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
                color="#94a3b8"
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.25),
                up=dict(x=0, y=0, z=1)
            )
        ),
        legend=dict(
            x=0.03,
            y=0.10,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=12, family="Inter, sans-serif"),
            itemsizing="constant"
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                x=0.03,
                y=0.88,
                showactive=True,
                bgcolor="rgba(30, 41, 59, 0.85)",
                bordercolor="#475569",
                font=dict(color="#f8fafc", size=11, family="Inter, sans-serif"),
                buttons=camera_presets
            )
        ]
    )

    # Save completely self-contained offline HTML
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    fig.write_html(
        output_html_path,
        include_plotlyjs=True,
        full_html=True,
        config=dict(
            displayModeBar=True,
            responsive=True,
            displaylogo=False
        )
    )
    return output_html_path


def create_static_3d_snapshot(
    patient_id,
    metrics,
    verts_pred=None, faces_pred=None,
    verts_gt=None, faces_gt=None,
    verts_brain=None, faces_brain=None,
    output_png_path="render_3d_snapshot.png"
):
    """
    Generates a 4-panel static 3D projection figure using Matplotlib:
      1. Isometric 3D View (Perspective)
      2. Superior View (Axial / Top-Down)
      3. Lateral View (Sagittal / Side)
      4. Anterior View (Coronal / Front)
    """
    fig = plt.figure(figsize=(18, 14), facecolor="#090d16")
    
    # Subsample faces for fast matplotlib rendering
    brain_coll_data = None
    if verts_brain is not None and faces_brain is not None:
        sub_faces = faces_brain[::8]  # Subsample brain for fast rendering
        brain_coll_data = verts_brain[sub_faces]

    gt_polys = verts_gt[faces_gt[::2]] if (verts_gt is not None and faces_gt is not None) else None
    pred_polys = verts_pred[faces_pred[::2]] if (verts_pred is not None and faces_pred is not None) else None

    views = [
        ("3D Perspective View", 25, 45),
        ("Superior View (Axial)", 90, -90),
        ("Lateral View (Sagittal)", 0, 0),
        ("Anterior View (Coronal)", 0, -90)
    ]

    for idx, (title, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d", facecolor="#090d16")
        
        # 1. Translucent Brain Shell
        if brain_coll_data is not None:
            mesh_brain = Poly3DCollection(brain_coll_data, alpha=0.04, edgecolor="none")
            mesh_brain.set_facecolor([0.5, 0.6, 0.7, 0.05])
            ax.add_collection3d(mesh_brain)

        # 2. Ground Truth (Teal)
        if gt_polys is not None:
            mesh_gt = Poly3DCollection(gt_polys, alpha=0.45, edgecolor="none")
            mesh_gt.set_facecolor([0.0, 0.95, 0.8, 0.5])
            ax.add_collection3d(mesh_gt)

        # 3. Predicted Tumor (Crimson)
        if pred_polys is not None:
            mesh_pred = Poly3DCollection(pred_polys, alpha=0.75, edgecolor="none")
            mesh_pred.set_facecolor([1.0, 0.0, 0.35, 0.8])
            ax.add_collection3d(mesh_pred)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
        
        # Coordinate limits & styling
        ax.set_xlim([0, 240])
        ax.set_ylim([0, 240])
        ax.set_zlim([0, 155])
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#1e293b")
        ax.yaxis.pane.set_edgecolor("#1e293b")
        ax.zaxis.pane.set_edgecolor("#1e293b")
        ax.grid(color="#1e293b", linestyle=":", linewidth=0.5)

    # Master Title & Metrics
    dice_val = f"{metrics.get('dice', 0)*100:.1f}%" if metrics and 'dice' in metrics else "N/A"
    iou_val = f"{metrics.get('iou', 0)*100:.1f}%" if metrics and 'iou' in metrics else "N/A"
    pred_vol = f"{metrics.get('pred_ml', 0):.1f} mL" if metrics else "N/A"
    gt_vol = f"{metrics.get('true_ml', 0):.1f} mL" if metrics else "N/A"

    plt.suptitle(
        f"3D Multi-Planar Volumetric Reconstruction — {patient_id}\n"
        f"Dice: {dice_val}  |  IoU: {iou_val}  |  Pred Volume: {pred_vol}  |  True Volume: {gt_vol}",
        color="#38bdf8", fontsize=16, fontweight="bold", y=0.98
    )

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff0055', markersize=10, label='Predicted Tumor (Carver + Sieve)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#00f5d4', markersize=10, label='Ground Truth Annotation'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markersize=10, label='Glass Brain Hull')
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=True,
               facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.savefig(output_png_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    return output_png_path


def render_patient_3d(patient_target, open_browser=False, skip_brain=False, save_png=True):
    """
    Main orchestration routine for 3D volumetric rendering.
    """
    p_id, flair_path, seg_path = find_patient_files(patient_target)
    if not flair_path or not os.path.exists(flair_path):
        raise FileNotFoundError(f"Could not locate FLAIR MRI for patient: '{patient_target}'")

    print("=" * 60)
    print(f"[3D RENDERER] Processing Patient : {p_id}")
    print(f"   FLAIR Path : {flair_path}")
    print(f"   SEG Path   : {seg_path}")
    print(f"   Device     : {DEVICE}")
    print("=" * 60)

    # 1. Run pipeline inference to get 3D predicted tumor mask and exact metrics
    pred_mask, metrics = run_contraption(flair_path, seg_path, p_id, mode="compare" if seg_path else "predict")

    # 2. Extract 3D Surface Meshes
    print("\n[3D RENDERER] Extracting 3D surface meshes via Marching Cubes...")
    
    # Predicted Tumor Mesh
    verts_pred, faces_pred = None, None
    if pred_mask is not None and np.sum(pred_mask) > 10:
        verts_pred, faces_pred, _ = extract_surface_mesh(
            pred_mask, level=0.5, step_size=1, smooth_sigma=0.5
        )
        print(f"  -> Predicted Tumor : {len(verts_pred):,} vertices | {len(faces_pred):,} triangles")
    else:
        print("  -> Predicted Tumor : No tumor voxels detected (empty mask)")

    # Ground Truth Tumor Mesh (if seg exists)
    verts_gt, faces_gt = None, None
    if seg_path and os.path.exists(seg_path):
        seg_vol = (nib.load(seg_path).get_fdata() > 0).astype(np.float32)
        if np.sum(seg_vol) > 10:
            verts_gt, faces_gt, _ = extract_surface_mesh(
                seg_vol, level=0.5, step_size=1, smooth_sigma=0.5
            )
            print(f"  -> Ground Truth    : {len(verts_gt):,} vertices | {len(faces_gt):,} triangles")

    # Brain Hull Mesh (Glass Brain)
    verts_brain, faces_brain = None, None
    if not skip_brain:
        flair_vol = nib.load(flair_path).get_fdata()
        verts_brain, faces_brain, _ = build_glass_brain_mesh(flair_vol, step_size=2)
        print(f"  -> Glass Brain     : {len(verts_brain):,} vertices | {len(faces_brain):,} triangles")

    # 3. Export Targets
    out_dir = os.path.join(OUTPUTS_DIR, p_id)
    os.makedirs(out_dir, exist_ok=True)
    
    html_path = os.path.join(out_dir, "render_3d.html")
    png_path = os.path.join(out_dir, "render_3d_snapshot.png")

    # 4. Generate Interactive 3D WebGL HTML
    print(f"\n[3D RENDERER] Generating interactive 3D WebGL HTML...")
    create_interactive_3d_html(
        patient_id=p_id,
        metrics=metrics,
        verts_pred=verts_pred, faces_pred=faces_pred,
        verts_gt=verts_gt, faces_gt=faces_gt,
        verts_brain=verts_brain, faces_brain=faces_brain,
        output_html_path=html_path
    )
    print(f"  -> Saved Interactive 3D HTML : {html_path}")

    # 5. Generate Static 4-Panel 3D Snapshot PNG
    if save_png:
        print(f"[3D RENDERER] Generating 4-panel static 3D perspective snapshot PNG...")
        create_static_3d_snapshot(
            patient_id=p_id,
            metrics=metrics,
            verts_pred=verts_pred, faces_pred=faces_pred,
            verts_gt=verts_gt, faces_gt=faces_gt,
            verts_brain=verts_brain, faces_brain=faces_brain,
            output_png_path=png_path
        )
        print(f"  -> Saved Static 3D Snapshot  : {png_path}")

    print("=" * 60)
    print("[3D RENDERER] SUCCESS: 3D Visualization Pipeline Complete!")
    print(f"  Interactive HTML : file:///{html_path.replace(os.sep, '/')}")
    if save_png:
        print(f"  Static Snapshot  : file:///{png_path.replace(os.sep, '/')}")
    print("=" * 60)

    # 6. Auto-open in browser if requested
    if open_browser:
        webbrowser.open(f"file:///{os.path.abspath(html_path)}")

    return html_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Volumetric Brain & Tumor Renderer")
    parser.add_argument("patient", nargs="?", default="BraTS20_Training_001",
                        help="Patient ID (e.g. BraTS20_Training_001) or direct path to flair.nii")
    parser.add_argument("--open", action="store_true", help="Automatically open generated 3D HTML in default browser")
    parser.add_argument("--skip-brain", action="store_true", help="Skip the glass brain hull and render only tumor meshes")
    parser.add_argument("--no-png", action="store_true", help="Skip generating the 4-panel static 3D snapshot PNG")
    args = parser.parse_args()

    render_patient_3d(
        patient_target=args.patient,
        open_browser=args.open,
        skip_brain=args.skip_brain,
        save_png=not args.no_png
    )
