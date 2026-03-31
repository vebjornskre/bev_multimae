import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_projection(img: np.ndarray, proj: dict, save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.imshow(img)
    sc = ax.scatter(proj["u"], proj["v"], c=proj["depth_cam"],
                    cmap="plasma_r", s=18, linewidths=0, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Radar depth (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'projection_viz'))
    plt.close()


def visualize_calibration_fit(d_pred: np.ndarray, d_radar: np.ndarray,
                               alpha: float, beta: float, save_path: str):
    finite = np.isfinite(d_pred) & np.isfinite(d_radar)
    d_pred, d_radar = d_pred[finite], d_radar[finite]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d_pred, d_radar, s=12, alpha=0.6)
    x = np.linspace(d_pred.min(), d_pred.max(), 100)
    ax.plot(x, alpha * x + beta, "r-", linewidth=2, zorder=5,
            label=f"α={alpha:.3f}  β={beta:.3f}")
    ax.set_xlabel("Model prediction (m)")
    ax.set_ylabel("Radar camera-frame Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'calibration_fit_viz'))
    plt.close()

def plot_depth_residuals(cfg, u, v, residuals, correction, H, W):
    save_folder = os.path.join(cfg.plot_folder, "depth")
    os.makedirs(save_folder, exist_ok=True)

    max_val = max(np.max(np.abs(residuals)), np.max(np.abs(correction)))
    vmin, vmax = -max_val, max_val

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(u, v, c=residuals, cmap='coolwarm', vmin=vmin, vmax=vmax, s=2)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, "residuals.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(correction, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, "correction.png"), dpi=150)
    plt.close(fig)