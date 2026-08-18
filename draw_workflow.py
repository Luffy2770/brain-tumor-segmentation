import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    bg_color = "#f8f9fa"
    c_blue = "#2b7bba"
    c_teal = "#1abc9c"
    c_orange = "#e67e22"
    c_green = "#2ecc71"
    c_red = "#e74c3c"
    c_purple = "#9b59b6"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Title
    ax.text(50, 95, "Cascaded Medical Image Analysis Pipeline", fontsize=18, fontweight='bold', ha='center', color="#2c3e50")
    ax.text(50, 92, "Brain Tumor Screening, 2D U-Net Segmentation & 3D Post-Processing", fontsize=12, ha='center', color="#7f8c8d")

    def add_box(x, y, w, h, title, subtitle, color, shape="rect"):
        if shape == "rect":
            box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                        facecolor=color, edgecolor="#2c3e50", linewidth=1.5, alpha=0.9)
            ax.add_patch(box)
        elif shape == "diamond":
            diamond = patches.Polygon([[x, y + h/2], [x + w/2, y], [x, y - h/2], [x - w/2, y]],
                                      facecolor=color, edgecolor="#2c3e50", linewidth=1.5, alpha=0.9)
            ax.add_patch(diamond)

        ax.text(x, y + (1.5 if subtitle else 0), title, fontsize=11, fontweight='bold', ha='center', va='center', color="white")
        if subtitle:
            ax.text(x, y - 2, subtitle, fontsize=9, ha='center', va='center', color="#f1f2f6")

    def add_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#34495e', edgecolor='#34495e', width=1.5, headwidth=7, shrink=0.08))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 3, my, label, fontsize=9, fontweight='bold', color="#c0392b" if "Yes" in label else "#27ae60", ha='left', va='center')

    # ---- Pipeline Nodes ----
    # 1. Input
    add_box(15, 80, 22, 10, "1. Input MRI Volume", "3D FLAIR NIfTI Scan (.nii)", c_blue)
    
    # 2. Preprocessing
    add_box(15, 62, 22, 10, "2. Preprocessing & Resizing", "Foreground Norm + 128x128", c_teal)
    add_arrow(15, 75, 15, 67)

    # 3. Stage 1 Screening
    add_box(15, 44, 22, 10, "3. Stage 1 Classifier", "TumorClassifierCNN (3-Layer)", c_orange)
    add_arrow(15, 57, 15, 49)

    # Decision Gate
    add_box(50, 44, 22, 10, "Tumor Prob > 0.35 ?", "Slice Threshold Check", "#34495e", shape="diamond")
    add_arrow(26, 44, 39, 44)

    # 4. Skip Clean
    add_box(50, 22, 22, 10, "Skip Heavy UNet", "Mark Slice 'No Tumor'", "#95a5a6")
    add_arrow(50, 39, 50, 27, label="No (Clean)")

    # 5. Stage 2 UNet
    add_box(85, 44, 22, 10, "4. Stage 2 Segmentation", "UNet2D + Bilinear Rescaling", c_green)
    add_arrow(61, 44, 74, 44, label="Yes (Tumor)")

    # 6. Stage 3 Cleanup
    add_box(85, 22, 22, 10, "5. 3D Fragment Cleanup", "scipy.ndimage connected components", c_red)
    add_arrow(85, 39, 85, 27)

    # 7. Outputs & Metrics
    add_box(50, 6, 40, 10, "6. Diagnostics, Timeline Barcode & Mask Evaluation", "Slice CSV + Barcode PNG + MIP Overlay + 3D Dice Evaluation", c_purple)
    add_arrow(85, 17, 70, 6)
    add_arrow(50, 17, 50, 11)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "processed", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "workflow_chart.png")
    plt.savefig(out_file, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close(fig)
    print(f"Flowchart successfully generated and saved to: {out_file}")

if __name__ == "__main__":
    draw_flowchart()
