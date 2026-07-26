import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_projection(img: np.ndarray, proj: dict, save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.imshow(img)

    sc = ax.scatter(
        proj["u"], proj["v"],
        c=proj["depth_cam"],
        cmap="plasma_r",
        s=18,
        linewidths=0,
        alpha=0.85,
    )

    ax.set_xlabel("u (px)", fontsize=26)
    ax.set_ylabel("v (px)", fontsize=26)
    ax.tick_params(axis="both", labelsize=26)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.35)

    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label("LiDAR depth [m]", fontsize=26)
    cbar.ax.tick_params(labelsize=26)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "projection_viz.png"), dpi=150)
    plt.close()


def visualize_calibration_fit(d_pred: np.ndarray, d_radar: np.ndarray,
                               alpha: float, beta: float, save_path: str):
    finite = np.isfinite(d_pred) & np.isfinite(d_radar)
    d_pred, d_radar = d_pred[finite], d_radar[finite]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("MoGe-2 RANSAC scale-bias fit")
    ax.scatter(d_pred, d_radar, s=12, alpha=0.6)
    x = np.linspace(d_pred.min(), d_pred.max(), 100)
    ax.plot(x, alpha * x + beta, "r-", linewidth=2, zorder=5,
            label=f"α={alpha:.3f}  β={beta:.3f}")
    ax.set_xlabel("MoGe-2 prediction (m)")
    ax.set_ylabel("LiDAR camera-frame Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'calibration_fit_viz'))
    plt.close()

def plot_depth_residuals(cfg, u, v, residuals, correction, H, W, img=None):
    save_folder = os.path.join(cfg.plot_folder, "depth")
    os.makedirs(save_folder, exist_ok=True)

    if img is not None:
        img_np = np.array(img)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(img_np)
        ax.set_xlabel("u (px)", fontsize=14)
        ax.set_ylabel("v (py)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, "image.png"), dpi=150)
        plt.close(fig)

    max_val = max(np.max(np.abs(residuals)), np.max(np.abs(correction)))
    vmin, vmax = -max_val, max_val

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(u, v, c=residuals, cmap="coolwarm", vmin=vmin, vmax=vmax, s=1.2)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u (px)", fontsize=11)
    ax.set_ylabel("v (px)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, aspect=10, pad=0.04)
    cbar.set_label("Residual (m)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, "residuals.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(correction, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)", fontsize=11)
    ax.set_ylabel("v (px)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.65, aspect=10, pad=0.04)
    cbar.set_label("Depth correction (m)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, "correction.png"), dpi=150)
    plt.close(fig)