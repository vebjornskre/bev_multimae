import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import os

import hydra
from omegaconf import DictConfig

from bev_multimae.preprocessing.camera.depth import DepthEstimator, load_single_img, cnn_feature_extract
from bev_multimae.visualization.camera_points_viz import plot_lifted_points

from bev_multimae.preprocessing.mcap_reader import get_camera_transform, get_radar_transform
from bev_multimae.preprocessing.get_transforms import T_rad_to_ego, T_cam_to_ego, apply_transform

from bev_multimae.preprocessing.camera.camera_depth_calibration import calibrate_depth_with_sensor

log = logging.getLogger(__name__)


def project_2D_3D(cfg, depth, T, img_size=None):
    cam_info = np.load(cfg.camera_info)
    K, D = cam_info['K'], cam_info['D']

    if isinstance(depth, torch.Tensor):
        depth_np = depth.squeeze().cpu().numpy()
    else:
        depth_np = np.squeeze(depth)

    H, W = depth_np.shape

    # Scale intrinsics to match depth map resolution
    if img_size is not None:
        W_orig, H_orig = img_size

        K = K.copy()
        K[0, :] *= W / W_orig
        K[1, :] *= H / H_orig

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    pixel_coords = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
    norm_coords = cv2.undistortPoints(pixel_coords, K, D).reshape(H, W, 2)

    # Mark pixels where undistortPoints iteration diverged
    bad = np.any(np.abs(norm_coords) > 2.0, axis=-1)
    bad = np.any(np.abs(norm_coords) > 10.0, axis=-1)
    norm_coords[bad] = np.nan

    rays = np.concatenate([norm_coords, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    points_cam = rays * (depth_np[..., np.newaxis] / rays[..., 2:3])

    # Transform points from camera frame to ego frame
    pts = points_cam.reshape(-1, 3)
    valid = (
        ~np.isnan(pts).any(axis=1) &
        np.isfinite(pts).all(axis=1)
    )
    pts[valid] = apply_transform(T, pts[valid])
    pts[~valid] = np.nan
    points_3d = pts.reshape(H, W, 3)

    # log.info(f"Lifted point cloud shape: {points_3d.shape}  "
    #          f"(X range [{np.nanmin(points_3d[...,0]):.2f}, {np.nanmax(points_3d[...,0]):.2f}], "
    #          f"Y range [{np.nanmin(points_3d[...,1]):.2f}, {np.nanmax(points_3d[...,1]):.2f}])")

    return torch.from_numpy(points_3d)

def lift(
        cfg, 
        img,
        cal_pts,
        de: DepthEstimator = None, 
        plot=False
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns lifted 3D points (N,3) and RGB colors (N,3) in ego frame.
    """
    
    depth = de._predict(img)

    from scipy.spatial.transform import Rotation

    # camera
    _T_cam_to_ego = T_cam_to_ego(cfg.mcap_path)
    
    # radar
    _T_rad_to_ego = T_rad_to_ego(cfg.mcap_path)

    if isinstance(img, torch.Tensor):
        _, H_orig, W_orig = img.shape
    elif isinstance(img, np.ndarray):
        H_orig, W_orig = img.shape[:2]
    else:
        W_orig, H_orig = img.size   

    img_size = (W_orig, H_orig)
    depth_np = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)

    cam_info = np.load(cfg.camera_info)
    K, D = cam_info['K'], cam_info['D']
    H_dep, W_dep = depth_np.shape
    fx, fy = K[0, 0] * (W_dep / W_orig), K[1, 1] * (H_dep / H_orig)
    cx, cy = K[0, 2] * (W_dep / W_orig), K[1, 2] * (H_dep / H_orig)
    u_grid, v_grid = np.meshgrid(np.arange(W_dep, dtype=np.float32), np.arange(H_dep, dtype=np.float32))
    ray_lengths = np.sqrt(((u_grid - cx) / fx)**2 + ((v_grid - cy) / fy)**2 + 1.0)
    depth_np = depth_np * ray_lengths

    alpha, beta = calibrate_depth_with_sensor(
        cfg, 
        depth_np, 
        img_hw=img_size[::-1], 
        depth_hw=depth_np.shape, 
        cal_pts=cal_pts,
        plot=True,
        img=img
        )

    depth = torch.from_numpy(alpha * depth_np + beta)

    ego_cam_pts = project_2D_3D(cfg, depth, _T_cam_to_ego, img_size=img_size)

    img_np = np.array(img, dtype=np.float32) / 255.0
    H, W = ego_cam_pts.shape[:2]
    if img_np.shape[0] != H or img_np.shape[1] != W:
        img_np = cv2.resize(img_np, (W, H))

    ego_cam_pts = ego_cam_pts.reshape(-1, 3).numpy()
    colors = img_np.reshape(-1, 3)

    if plot:
        plot_lifted_points(cfg, ego_cam_pts, colors, img, meshlab=True)

    valid = (
        ~np.isnan(ego_cam_pts).any(axis=1) &
        np.isfinite(ego_cam_pts).all(axis=1)
    )
    return ego_cam_pts[valid], colors[valid]

@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    print('This files shoulnd be run at this time')

if __name__ == '__main__':
    main()
