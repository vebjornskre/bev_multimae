import os
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from bev_multimae.preprocessing.get_transforms import apply_transform, T_cam_to_ego


def _normalize_patch_size(patch_size_pixels):
    if isinstance(patch_size_pixels, (tuple, list)):
        patch_h_px, patch_w_px = patch_size_pixels
    else:
        patch_h_px = patch_w_px = patch_size_pixels
    return patch_h_px, patch_w_px

def plot_bev_comparison(cfg, img, pts_radar_ego, bev_cam_hires, voxel_size, point_cloud_range, patch_size_pixels, i):
    save_folder = os.path.join(cfg.plot_folder, "BEV")
    os.makedirs(save_folder, exist_ok=True)

    bev_hi_np = bev_cam_hires.permute(1, 2, 0).numpy()

    print(f'bev cam hires shape: {bev_cam_hires.shape}')
    print(f'bev cam hires numpy shape: {bev_hi_np.shape}')

    pcr = point_cloud_range
    x_min, y_min, x_max, y_max = pcr[0], pcr[1], pcr[3], pcr[4]
    x_range = x_max - x_min
    y_range = y_max - y_min
    extent = [0, x_range, 0, y_range]

    # Compute patch size in metric space
    bev_h_px, bev_w_px = bev_hi_np.shape[:2]
    px_per_m_x = bev_w_px / x_range
    px_per_m_y = bev_h_px / y_range
    patch_h_px, patch_w_px = _normalize_patch_size(patch_size_pixels)
    patch_step_x_m = patch_w_px / px_per_m_x
    patch_step_y_m = patch_h_px / px_per_m_y

    def apply_ticks(ax):
        x_metric = np.arange(x_min, x_max + 1, 10)
        y_metric = np.arange(y_min, y_max + 1, 10)
        ax.set_xticks(x_metric - x_min)
        ax.set_xticklabels([f"{v:.0f}m" for v in x_metric])
        ax.set_yticks(y_metric - y_min)
        ax.set_yticklabels([f"{v:.0f}m" for v in y_metric])
        ax.set_xlabel("x / forward (m)")
        ax.set_ylabel("y / lateral (m)")

    def draw_patch_grid(ax, step_x_m, step_y_m, color, lw=0.8):
        v_lines = [[(x, 0), (x, y_range)] for x in np.arange(0, x_range + step_x_m, step_x_m)]
        h_lines = [[(0, y), (x_range, y)] for y in np.arange(0, y_range + step_y_m, step_y_m)]
        ax.add_collection(LineCollection(v_lines + h_lines, colors=color, linewidths=lw, alpha=0.8, zorder=2))

    height = pts_radar_ego[:, 2]
    valid = height > -3
    pts_radar_ego = pts_radar_ego[valid]

    px_rad = pts_radar_ego[:, 0] - x_min
    py_rad = pts_radar_ego[:, 1] - y_min

    fig_overlay, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax = axes[0]
    ax.imshow(bev_hi_np, origin="lower", aspect="auto", extent=extent)
    draw_patch_grid(ax, patch_step_x_m, patch_step_y_m, color="red")

    ax.scatter(
        px_rad, py_rad,
        s=20,
        c="lime",
        alpha=1.0,
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
        label="Radar points"
    )

    ax.set_xlim(0, x_range)
    ax.set_ylim(0, y_range)
    apply_ticks(ax)
    ax.set_title("Camera BEV + Radar Overlay")
    ax.legend(fontsize=8)

    ax_img = axes[1]
    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = np.array(img)
    ax_img.imshow(img_np)
    ax_img.set_title("Image")
    ax_img.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"video/bev_overlay_{i}.png"), dpi=150)
    plt.close(fig_overlay)


def overlay_radar_on_image(cfg, img, pts_rad_ego, T_cam_ego):
    cam_info = np.load(cfg.camera_info)
    K, D = cam_info['K'], cam_info['D']
    img_np = np.array(img)
    img_hw = img_np.shape[:2]

    T_ego_cam = np.linalg.inv(T_cam_ego)
    pts_xyz = pts_rad_ego[:, :3]
    pts_rad_camFrame = apply_transform(T_ego_cam, pts_xyz)  

    valid = pts_rad_camFrame[:, 2] > 0
    pts_rad_camFrame = pts_rad_camFrame[valid]
    depths = pts_rad_camFrame[:, 2]
    
    uv, _ = cv2.projectPoints(
        pts_rad_camFrame.astype(np.float64),
        np.zeros(3), np.zeros(3),
        K.astype(np.float64), D
    )

    uv = uv.reshape(-1, 2)
    
    H, W = img_np.shape[:2]
    inside = (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
    uv = uv[inside].astype(int)
    depths = depths[inside]

    d_min, d_max = depths.min(), depths.max()
    for (u, v), d in zip(uv, depths):
        intensity = int(255 * (1.0 - (d - d_min) / (d_max - d_min)))
        color = (0, intensity, 0)
        cv2.circle(img_np, (u, v), 5, color, -1)
    
    return img_np

def save_ply_color(fname, pts, color):
    pts = np.asarray(pts)

    assert pts.shape[1] == 3, "Points must be (N, 3)"

    with open(fname, "w") as f:
        # header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        r, g, b = color

        # points
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")


def plot_bev_target(cfg, bev, name="bev_target"):
    save_folder = os.path.join(cfg.plot_folder, "BEV_target")
    os.makedirs(save_folder, exist_ok=True)

    bev = bev[0].detach().cpu().numpy()
    C = bev.shape[0]
    print(f'radar bev shape: {bev.shape}')

    titles = [
        "Occupancy",
        "Density (log count)",
        "Height (mean z)",
        "Velocity (mean)",
        "RCS (mean)",
        "Height (var)",
        "Velocity (var)",
        "RCS (var)",
        "SNR (mean)",
    ]

    cols = 3
    rows = int(np.ceil(C / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = np.array(axs).reshape(-1)

    for i in range(C):
        ch = bev[i]

        if i == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.min(ch))
            vmax = float(np.max(ch))
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6

        im = axs[i].imshow(ch, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axs[i].set_title(titles[i] if i < len(titles) else f"Channel {i}")
        fig.colorbar(im, ax=axs[i], fraction=0.046)
        axs[i].set_xlabel("Forward")
        axs[i].set_ylabel("Left")

    for i in range(C, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()

    save_path = os.path.join(save_folder, f"{name}.png")
    plt.savefig(save_path)
    plt.close()