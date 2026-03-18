import numpy as np
import cv2
import torch
import logging
import matplotlib.pyplot as plt
import os

import hydra
from omegaconf import DictConfig

from bev_multimae.visualization.depth_visualization import plot_depth_maps
from bev_multimae.preprocessing.radar.radar_process_utils import load_and_process, build_thresholds
from bev_multimae.preprocessing.mcap_reader import (
    get_radar_to_camera_transform,
    apply_transform,
)
from bev_multimae.preprocessing.camera.depth import DepthEstimator, load_single_img

log = logging.getLogger(__name__)


def project_radar_to_image(radar: dict, T_radar_to_cam: np.ndarray, K: np.ndarray,
                           D: np.ndarray, img_hw: tuple, depth_hw: tuple) -> dict:
    H_img, W_img = img_hw
    H_dep, W_dep = depth_hw

    pts_radar = np.stack([radar["x"], radar["y"], radar["z"]], axis=-1)
    pts_cam   = apply_transform(T_radar_to_cam, pts_radar)

    log.info(f"pts_cam Z: min={pts_cam[:,2].min():.2f}  max={pts_cam[:,2].max():.2f}  mean={pts_cam[:,2].mean():.2f}")

    uv, _ = cv2.projectPoints(
        pts_cam.astype(np.float64),
        np.zeros(3), np.zeros(3),
        K.astype(np.float64), D.astype(np.float64),
    )
    uv = uv.reshape(-1, 2)

    u_dep = (uv[:, 0] * (W_dep / W_img)).astype(int)
    v_dep = (uv[:, 1] * (H_dep / H_img)).astype(int)

    inside = (u_dep >= 0) & (u_dep < W_dep) & (v_dep >= 0) & (v_dep < H_dep)
    log.info(f"After image bounds: {inside.sum()}")

    return {
        "u":           u_dep[inside],
        "v":           v_dep[inside],
        "depth_cam":   pts_cam[inside, 2],
        "radial_dist": radar["radial_distance"][inside],
        "snr":         radar["signal_noise_ratio"][inside],
        "rcs":         radar["radar_cross_section"][inside],
        "elevation":   radar["elevation_angle"][inside],
    }


def fit_depth_scale(depth_map: np.ndarray, proj: dict,
                    use_ransac: bool = True) -> tuple[float, float]:
    u, v    = proj["u"], proj["v"]
    d_pred  = depth_map[v, u].astype(np.float64)
    d_radar = proj["depth_cam"].astype(np.float64)

    valid  = d_pred > 0.01
    d_pred, d_radar = d_pred[valid], d_radar[valid]

    if len(d_pred) < 4:
        raise ValueError(f"Too few valid correspondences: {len(d_pred)}")

    log.info(f"Fitting on {len(d_pred)} correspondences")

    if use_ransac:
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        ransac = RANSACRegressor(
            LinearRegression(),
            residual_threshold=0.3,
            min_samples=4,
            random_state=42,
        )
        ransac.fit(d_pred.reshape(-1, 1), d_radar)
        alpha = float(ransac.estimator_.coef_[0])
        beta  = float(ransac.estimator_.intercept_)
        log.info(f"RANSAC inlier ratio: {ransac.inlier_mask_.mean():.2f}")
    else:
        A = np.stack([d_pred, np.ones_like(d_pred)], axis=1)
        (alpha, beta), _, _, _ = np.linalg.lstsq(A, d_radar, rcond=None)
        alpha, beta = float(alpha), float(beta)

    log.info(f"alpha={alpha:.4f}  beta={beta:.4f}")
    return alpha, beta


def calibrate_depth(depth_map: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.clip(alpha * depth_map + beta, 0, None)


def visualize_projection(img: np.ndarray, proj: dict, save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.imshow(img)
    sc = ax.scatter(proj["u"], proj["v"], c=proj["depth_cam"],
                    cmap="plasma_r", s=18, linewidths=0, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Radar depth (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'projection_viz'))
    log.info('Saved projection visualization as projection_viz')
    plt.close()

def visualize_calibration_fit(d_pred: np.ndarray, d_radar: np.ndarray,
                               alpha: float, beta: float, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d_pred, d_radar, s=12, alpha=0.6)
    x = np.linspace(d_pred.min(), d_pred.max(), 100)
    ax.plot(x, alpha * x + beta, "r-", linewidth=2,
            label=f"α={alpha:.3f}  β={beta:.3f}")
    ax.set_xlabel("Model prediction (m)")
    ax.set_ylabel("Radar camera-frame Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'calibration_fit_viz'))
    log.info('Saved calibration fit image as calibration_fit_viz')
    plt.close()

def calibrate_depth_with_radar(cfg, depth_np: np.ndarray, img_hw: tuple, depth_hw: tuple) -> tuple[float, float]:
    cam_info = np.load(cfg.camera_info)
    K, D     = cam_info["K"], cam_info["D"]

    thresholds     = build_thresholds(cfg)
    radar          = load_and_process(cfg, thresholds)
    T_radar_cam = get_radar_to_camera_transform(cfg.mcap_path)

    proj = project_radar_to_image(radar, T_radar_cam, K, D, img_hw=img_hw, depth_hw=depth_hw)

    alpha, beta = fit_depth_scale(depth_np, proj, use_ransac=True)
    return alpha, beta


@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Camera
    img   = load_single_img(cfg)
    de    = DepthEstimator(cfg, device, plot=False)
    de._load_model()
    depth = de._predict(img)

    depth_np     = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)
    H_dep, W_dep = depth_np.shape
    img_np       = np.array(img)
    H_img, W_img = img_np.shape[:2]

    cam_info = np.load(cfg.camera_info)
    K, D     = cam_info["K"], cam_info["D"]

    # Radar
    thresholds      = build_thresholds(cfg)
    radar           = load_and_process(cfg, thresholds)

    T_radar_to_cam = get_radar_to_camera_transform(cfg.mcap_path)
    proj = project_radar_to_image(
        radar, T_radar_to_cam, K, D,
        img_hw=(H_img, W_img),
        depth_hw=(H_dep, W_dep),
    )
    log.info(f"Projected {len(proj['u'])} radar points into image")

    visualize_projection(cv2.resize(img_np, (W_dep, H_dep)), proj, cfg.plot_folder)

    alpha, beta = fit_depth_scale(depth_np, proj, use_ransac=True)

    u, v   = proj["u"], proj["v"]
    d_pred = depth_np[v, u]
    valid  = d_pred > 0.01
    visualize_calibration_fit(d_pred[valid], proj["depth_cam"][valid], alpha, beta, cfg.plot_folder)

    depth_calibrated = calibrate_depth(depth_np, alpha, beta)
    log.info(f"Calibrated depth: min={depth_calibrated.min():.2f}  "
             f"max={depth_calibrated.max():.2f}  mean={depth_calibrated.mean():.2f}")

    plot_depth_maps(cfg, img, depth_calibrated)

if __name__ == "__main__":
    main()