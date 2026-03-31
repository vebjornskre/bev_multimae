import torch
import numpy as np
import cv2
import logging
from PIL import Image

import hydra
from omegaconf import DictConfig

from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.visualization.camera_points_viz import plot_lifted_points
from bev_multimae.preprocessing.get_transforms import T_rad_to_ego, T_cam_to_ego, apply_transform
from bev_multimae.preprocessing.camera.camera_depth_calibration import calibrate_depth_with_sensor

log = logging.getLogger(__name__)


def project_2D_3D(cfg, depth, T, K, img_size=None):
    if isinstance(depth, torch.Tensor):
        depth_np = depth.squeeze().cpu().numpy()
    else:
        depth_np = np.squeeze(depth)

    H, W = depth_np.shape

    if img_size is not None:
        W_orig, H_orig = img_size
        K = K.copy()
        K[0, :] *= W / W_orig
        K[1, :] *= H / H_orig

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pixel_coords = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
    norm_coords = cv2.undistortPoints(pixel_coords, K, None).reshape(H, W, 2)

    rays = np.concatenate([norm_coords, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    points_cam = rays * (depth_np[..., np.newaxis] / rays[..., 2:3])

    pts = points_cam.reshape(-1, 3)
    valid = ~np.isnan(pts).any(axis=1) & np.isfinite(pts).all(axis=1)
    pts[valid] = apply_transform(T, pts[valid])
    pts[~valid] = np.nan

    return torch.from_numpy(pts.reshape(H, W, 3))


def lift(cfg, img, cal_pts, de: DepthEstimator = None, plot=False) -> tuple[np.ndarray, np.ndarray]:
    """Returns lifted 3D points (N,3) and RGB colors (N,3) in ego frame."""
    cam_info = np.load(cfg.camera_info)
    K, D = cam_info['K'], cam_info['D']

    depth = de._predict(img)

    img_np = np.array(img)
    H, W = img_np.shape[:2]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0)
    img_size = (W, H)

    _T_cam_to_ego = T_cam_to_ego(cfg.mcap_path)

    depth_np = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)

    cam_info = np.load(cfg.camera_info) 
    K, D = cam_info['K'], cam_info['D'] 
    H_dep, W_dep = img_size

    depth_completed = calibrate_depth_with_sensor(
        cfg,
        depth_np,
        img_hw=(H, W),
        depth_hw=depth_np.shape,
        cal_pts=cal_pts,
        plot=True,
        img=img
    )

    depth = torch.from_numpy(depth_completed)
    ego_cam_pts = project_2D_3D(cfg, depth, _T_cam_to_ego, K_new, img_size=img_size)

    img_np = img_np.astype(np.float32) / 255.0
    H, W = ego_cam_pts.shape[:2]
    if img_np.shape[0] != H or img_np.shape[1] != W:
        img_np = cv2.resize(img_np, (W, H))

    ego_cam_pts = ego_cam_pts.reshape(-1, 3).numpy()
    colors = img_np.reshape(-1, 3)

    if plot:
        plot_lifted_points(cfg, ego_cam_pts, colors, img, meshlab=True)

    valid = ~np.isnan(ego_cam_pts).any(axis=1) & np.isfinite(ego_cam_pts).all(axis=1)
    return ego_cam_pts[valid], colors[valid]


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print('This files shoulnd be run at this time')

if __name__ == '__main__':
    main()