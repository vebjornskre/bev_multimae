import numpy as np
import cv2
import torch
import logging
import matplotlib.pyplot as plt
import os
from scipy.ndimage import map_coordinates
from sklearn.linear_model import RANSACRegressor, LinearRegression

import hydra
from omegaconf import DictConfig

from bev_multimae.visualization.depth_visualization import plot_depth_maps
from bev_multimae.preprocessing.mcap_reader import apply_transform
from bev_multimae.preprocessing.get_transforms import T_lid_to_cam, T_rad_to_cam
from bev_multimae.preprocessing.camera.depth import DepthEstimator, load_single_img

log = logging.getLogger(__name__)

def project_points_to_image(sensor, pts, T, K, D, img_hw, depth_hw):

    H_img, W_img = img_hw
    H_dep, W_dep = depth_hw

    if sensor == "radar":
        pts_xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1)
    else:
        pts_xyz = pts

    pts_cam = apply_transform(T, pts_xyz)

    valid_z = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid_z]

    if sensor == "radar":
        radar = {k: v[valid_z] for k, v in pts.items()}

    uv, _ = cv2.projectPoints(
        pts_cam.astype(np.float64),
        np.zeros(3), np.zeros(3),
        K.astype(np.float64), D.astype(np.float64),
    )
    uv = uv.reshape(-1, 2)

    u = uv[:, 0] * (W_dep / W_img)
    v = uv[:, 1] * (H_dep / H_img)

    inside = (
        (u >= 0) & (u < W_dep - 1) &
        (v >= 0) & (v < H_dep - 1)
    )

    out = {
        "u": u[inside],
        "v": v[inside],
        "depth_cam": pts_cam[inside, 2],
    }

    if sensor == "radar":
        out.update({
            "radial_dist": radar["radial_distance"][inside],
            "snr": radar["signal_noise_ratio"][inside],
            "rcs": radar["radar_cross_section"][inside],
            "elevation": radar["elevation_angle"][inside],
        })

    return out


def fit_depth_scale(cfg: DictConfig, depth_map: np.ndarray, proj: dict,
                    use_ransac: bool = True) -> tuple[float, float]:
    
    u, v    = proj["u"], proj["v"]

    # We use bilinear interpolation so we can sample depth at float (u, v)
    # instead of rounding to pixels, which makes the calibration smoother.
    coords = np.vstack([v, u])
    d_pred = map_coordinates(
        depth_map,
        coords, 
        order=1, # bilinnear
        mode='nearest'
        )

    d_radar = proj["depth_cam"].astype(np.float64)

    valid = (d_pred > 0.01) & np.isfinite(d_pred) & np.isfinite(d_radar)
    d_pred, d_radar = d_pred[valid], d_radar[valid]

    if len(d_pred) < 4:
        raise ValueError(f"Too few valid correspondences: {len(d_pred)}")
    
    if use_ransac:
        ransac = RANSACRegressor(
            LinearRegression(fit_intercept=cfg.fit_beta),
            residual_threshold=cfg.ransac_residual_threshold,
            min_samples=4,
            random_state=42,
        )
        ransac.fit(d_pred.reshape(-1, 1), d_radar)
        alpha = float(ransac.estimator_.coef_[0])

        if cfg.fit_beta == 0: beta = float(ransac.estimator_.intercept_)
        else: beta  = 0.0

    else:
        A = np.stack([d_pred, np.ones_like(d_pred)], axis=1)
        (alpha, beta), _, _, _ = np.linalg.lstsq(A, d_radar, rcond=None)
        alpha, beta = float(alpha), float(beta)


    return alpha, beta


def apply_calibration(depth_map: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    calibrated = np.clip(alpha * depth_map + beta, 0, None)
    calibrated[~np.isfinite(calibrated)] = np.nan
    return calibrated


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

def calibrate_depth_with_sensor(cfg, depth_np, img_hw, depth_hw, cal_pts, plot=False, img=None):
   
    cam_info = np.load(cfg.camera_info)
    K, D = cam_info["K"], cam_info["D"]
    
    if cfg.calibration == 'lidar': T = T_lid_to_cam(cfg.mcap_path)
    elif cfg.calibtation == 'radar': T = T_rad_to_cam(cfg.mcap_path)
    else: raise RuntimeError('Set calibration to either "lidar" or "radar"')

    proj = project_points_to_image(cfg.calibration, cal_pts, T, K, D, img_hw=img_hw, depth_hw=depth_hw)

    alpha, beta = fit_depth_scale(cfg, depth_np, proj, use_ransac=True)

    if plot and img is not None:
        u, v   = proj["u"], proj["v"]

        coords = np.vstack([v, u])
        d_pred = map_coordinates(depth_np, coords, order=1, mode='nearest')

        valid  = d_pred > 0.01
        H_dep, W_dep = depth_hw

        img_np = np.array(img)
        visualize_projection(cv2.resize(img_np, (W_dep, H_dep)), proj, cfg.plot_folder)
        visualize_calibration_fit(d_pred[valid], proj["depth_cam"][valid], alpha, beta, cfg.plot_folder)

    return alpha, beta


@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img   = load_single_img(cfg)
    de    = DepthEstimator(cfg, device, plot=False)
    de._load_model()
    depth = de._predict(img)

    depth_np     = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)
    H_dep, W_dep = depth_np.shape
    img_np       = np.array(img)
    H_img, W_img = img_np.shape[:2]

    alpha, beta = calibrate_depth_with_sensor(
        cfg, depth_np,
        img_hw=(H_img, W_img),
        depth_hw=(H_dep, W_dep),
        plot=True,
        img=img
    )

    depth_calibrated = apply_calibration(depth_np, alpha, beta)

    plot_depth_maps(cfg, img, depth_calibrated)

if __name__ == "__main__":
    main()