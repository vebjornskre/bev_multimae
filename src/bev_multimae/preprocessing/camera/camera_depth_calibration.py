import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
import os
from scipy.ndimage import map_coordinates
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.interpolate import RBFInterpolator

import hydra
from omegaconf import DictConfig

from bev_multimae.preprocessing.mcap_reader import apply_transform
from bev_multimae.preprocessing.get_transforms import T_lid_to_cam, T_rad_to_cam
from bev_multimae.visualization.depth_calibration_viz import plot_depth_residuals, visualize_calibration_fit, visualize_projection

log = logging.getLogger(__name__)


# def project_points_to_image(sensor, pts, T, K, D=None, img_hw=None, depth_hw=None):
#     H_img, W_img = img_hw
#     H_dep, W_dep = depth_hw

#     if sensor == "radar":
#         pts_xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1)
#     else:
#         pts_xyz = pts

#     pts_cam = apply_transform(T, pts_xyz)

#     log.info(f'min to max pts_cam[:,0]: {min(pts_cam[:,0]), max(pts_cam[:,0])}')
#     log.info(f'min to max pts_cam[:,1]: {min(pts_cam[:,1]), max(pts_cam[:,1])}')
#     log.info(f'min to max pts_cam[:,2]: {min(pts_cam[:,2]), max(pts_cam[:,2])}')

#     depth = pts_cam[:, 2]
#     height = pts_cam[:, 1]

#     valid_z = (depth > 1.0) & (depth < 60.0)
#     valid_y = (height > -3)

#     valid = valid_z & valid_y
#     pts_cam = pts_cam[valid]

#     if sensor == "radar":
#         radar = {k: v[valid] for k, v in pts.items()}

#     uv, _ = cv2.projectPoints(
#         pts_cam.astype(np.float64),
#         np.zeros(3), np.zeros(3),
#         K.astype(np.float64), D.astype(np.float64) if D is not None else None,
#     )
#     uv = uv.reshape(-1, 2)

#     u, v = uv[:, 0], uv[:, 1]
#     inside = (u >= 0) & (u < W_dep - 1) & (v >= 0) & (v < H_dep - 1)

#     out = {
#         "u": u[inside],
#         "v": v[inside],
#         "depth_cam": pts_cam[inside, 2],
#     }

#     if sensor == "radar":
#         out.update({
#             "radial_dist": radar["radial_distance"][inside],
#             "snr": radar["signal_noise_ratio"][inside],
#             "rcs": radar["radar_cross_section"][inside],
#             "elevation": radar["elevation_angle"][inside],
#         })

#     return out

def project_points_to_image(sensor, pts, T, K, D=None, img_hw=None, depth_hw=None):
    H_img, W_img = img_hw
    H_dep, W_dep = depth_hw

    if sensor == "radar":
        pts_xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1)
    else:
        pts_xyz = pts

    pts_cam = apply_transform(T, pts_xyz)

    depth = pts_cam[:, 2]
    height = pts_cam[:, 1]

    valid = (depth > 1.0) & (depth < 60.0) & (height > -3.0)
    pts_cam = pts_cam[valid]

    if sensor == "radar":
        radar = {k: v[valid] for k, v in pts.items()}

    sx = W_dep / W_img
    sy = H_dep / H_img

    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy

    uv, _ = cv2.projectPoints(
        pts_cam.astype(np.float64),
        np.zeros(3), np.zeros(3),
        K_scaled.astype(np.float64),
        None
    )
    uv = uv.reshape(-1, 2)

    u, v = uv[:, 0], uv[:, 1]

    inside = (u >= 0) & (u < W_dep - 1) & (v >= 0) & (v < H_dep - 1)

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
    u, v = proj["u"], proj["v"]

    d_pred = map_coordinates(depth_map, np.vstack([v, u]), order=1, mode='nearest')
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
        beta = float(ransac.estimator_.intercept_) if cfg.fit_beta else 0.0
    else:
        A = np.stack([d_pred, np.ones_like(d_pred)], axis=1)
        (alpha, beta), _, _, _ = np.linalg.lstsq(A, d_radar, rcond=None)
        alpha, beta = float(alpha), float(beta)

    return alpha, beta


def apply_calibration(depth_map: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    calibrated = np.clip(alpha * depth_map + beta, 0, None)
    calibrated[~np.isfinite(calibrated)] = np.nan
    return calibrated


def calibrate_depth_with_sensor(cfg, depth_np, img_hw, depth_hw, cal_pts, plot=False, img=None):
    cam_info = np.load(cfg.camera_info)
    K, D = cam_info["K"], cam_info["D"]

    if cfg.calibration == 'lidar': T = T_lid_to_cam(cfg.mcap_path)
    elif cfg.calibration == 'radar': T = T_rad_to_cam(cfg.mcap_path)
    else: raise RuntimeError('Set calibration to either "lidar" or "radar"')

    proj = project_points_to_image(
        cfg.calibration, cal_pts, T, K,
        D=D,
        img_hw=img_hw,
        depth_hw=depth_hw
    )

    log.info(f"Calibration depth range: min={proj['depth_cam'].min():.1f}m  max={proj['depth_cam'].max():.1f}m  median={np.median(proj['depth_cam']):.1f}m")

    if cfg.interp_depth_residuals:
        depth_cal = interp_depth_residuals(depth_np, proj, cfg, plot=True)

        log.info('Calibrating with interpolation')

        if plot and img is not None:
            H_dep, W_dep = depth_hw
            img_np = np.array(img)
            visualize_projection(cv2.resize(img_np, (W_dep, H_dep)), proj, cfg.plot_folder)

        return depth_cal
    
    log.info('Calibrating with alhpa and beta')

    alpha, beta = fit_depth_scale(cfg, depth_np, proj, use_ransac=True)

    # log.info(f'Aplha: {alpha}, Beat: {beta}')

    depth_cal = depth_np * alpha + beta

    if plot and img is not None:
        u, v = proj["u"], proj["v"]
        d_pred = map_coordinates(depth_np, np.vstack([v, u]), order=1, mode='nearest')
        valid = d_pred > 0.01
        H_dep, W_dep = depth_hw
        img_np = np.array(img)
        visualize_projection(cv2.resize(img_np, (W_dep, H_dep)), proj, cfg.plot_folder)
        visualize_calibration_fit(d_pred[valid], proj["depth_cam"][valid], alpha, beta, cfg.plot_folder)

    return depth_cal


def interp_depth_residuals(depth_np: np.ndarray, proj: dict, cfg, plot=False) -> np.ndarray:
    H, W = depth_np.shape
    u, v = proj["u"], proj["v"]

    d_moge_at_pts = map_coordinates(depth_np, np.vstack([v, u]), order=1, mode='nearest')

    valid = (
        (d_moge_at_pts > 0.01) &
        np.isfinite(d_moge_at_pts) &
        np.isfinite(proj["depth_cam"])
    )
    u, v = u[valid], v[valid]
    d_lidar = proj["depth_cam"][valid].astype(np.float64)
    d_moge_at_pts = d_moge_at_pts[valid]

    if len(u) < 4:
        log.warning("Too few valid lidar points for depth completion, returning raw depth")
        return depth_np.astype(np.float32)

    residuals = d_lidar - d_moge_at_pts

    mask = np.abs(residuals) < 6.0
    u, v = u[mask], v[mask]
    residuals = residuals[mask]

    pts_norm = np.stack([v / H, u / W], axis=1)
    rbf = RBFInterpolator(pts_norm, residuals, kernel='thin_plate_spline', smoothing=5)

    ug, vg = np.meshgrid(np.arange(W), np.arange(H))
    grid_norm = np.stack([vg.ravel() / H, ug.ravel() / W], axis=1)
    correction = rbf(grid_norm).reshape(H, W)

    correction = np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)

    if plot:
        plot_depth_residuals(cfg, u, v, residuals, correction, H, W)

    return np.clip(depth_np + correction.astype(np.float32), 0.1, None).astype(np.float32)


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print('Script does nothing on its own')

if __name__ == "__main__":
    main()