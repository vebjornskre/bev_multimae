import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

DEPTH_DIR = "reports/figures/depth_imgs"
MODELS = ['metric3d', 'depth_any', 'moge', 'depth_any_rel', 'depth_pro', 'unidepth', 'zoe']

def plot_depth_comparison(depth_dir, models):
    available = [m for m in models if os.path.exists(os.path.join(depth_dir, f"depth_map_{m}.npy"))]
    print(f"Found models: {available}")

    ncols = 4
    n = len(available) + 1  # +1 for RGB
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    # RGB in first cell
    rgb = Image.open(os.path.join(depth_dir, "rgb.png"))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=12, fontweight='bold', pad=8)
    axes[0].axis("off")
    # Invisible colorbar to align with depth plots
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cax.axis("off")

    # Depth maps in remaining cells
    for i, model in enumerate(available):
        ax = axes[i + 1]
        depth = np.load(os.path.join(depth_dir, f"depth_map_{model}.npy"))
        im = ax.imshow(depth, cmap="plasma")
        ax.set_title(model, fontsize=12, fontweight='bold', pad=8)
        ax.axis("off")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation="horizontal", label="depth (m)")

    # Hide any unused cells
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Depth Model Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(depth_dir, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    plot_depth_comparison(DEPTH_DIR, MODELS)