import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

import hydra
from omegaconf import DictConfig

from bev_multimae.preprocessing.camera.depth import DepthEstimator, load_single_img, cnn_feature_extract
from bev_multimae.visualization.camera_points_viz import plot_lifted_points
from bev_multimae.preprocessing.mcap_reader import get_camera_transform, apply_transform
from bev_multimae.preprocessing.camera.camera_depth_calibration import calibrate_depth_with_radar

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

    # Mark pixels where undistortPoints iteration diverged (known OpenCV issue)
    bad = np.any(np.abs(norm_coords) > 2.0, axis=-1)
    norm_coords[bad] = np.nan

    rays = np.concatenate([norm_coords, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    points_cam = rays * depth_np[..., np.newaxis]

    # Transform points from camera frame to base_link
    pts = points_cam.reshape(-1, 3)
    valid = ~np.isnan(pts).any(axis=1)
    pts[valid] = apply_transform(T, pts[valid])
    pts[~valid] = np.nan
    points_3d = pts.reshape(H, W, 3)

    # mask = (
    #     (points_3d[..., 0] >= 0) & (points_3d[..., 0] <= 80) &
    #     (points_3d[..., 1] >= -40) & (points_3d[..., 1] <= 40)
    # )
    # points_3d[~mask] = np.nan

    log.info(f"Lifted point cloud shape: {points_3d.shape}  "
             f"(X range [{np.nanmin(points_3d[...,0]):.2f}, {np.nanmax(points_3d[...,0]):.2f}], "
             f"Y range [{np.nanmin(points_3d[...,1]):.2f}, {np.nanmax(points_3d[...,1]):.2f}])")

    return torch.from_numpy(points_3d)

def load_and_lift(cfg, device: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns lifted 3D points (N,3) and RGB colors (N,3) in ego frame.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = load_single_img(cfg)

    de = DepthEstimator(cfg, device, plot=False)
    de._load_model()
    depth = de._predict(img)

    T = get_camera_transform(cfg.mcap_path)

    if isinstance(img, torch.Tensor):
        _, H_orig, W_orig = img.shape
    elif isinstance(img, np.ndarray):
        H_orig, W_orig = img.shape[:2]
    else:
        W_orig, H_orig = img.size

    img_size = (W_orig, H_orig)
    depth_np = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)

    alpha, beta = calibrate_depth_with_radar(cfg, depth_np, img_hw=img_size[::-1], depth_hw=depth_np.shape)
    depth = torch.from_numpy(np.clip(alpha * depth_np + beta, 0, None)).to(device)

    points = project_2D_3D(cfg, depth, T, img_size=img_size)

    img_np = np.array(img, dtype=np.float32) / 255.0
    H, W = points.shape[:2]
    if img_np.shape[0] != H or img_np.shape[1] != W:
        img_np = cv2.resize(img_np, (W, H))

    pts = points.reshape(-1, 3).numpy()
    colors = img_np.reshape(-1, 3)

    # remove NaN points
    valid = ~np.isnan(pts).any(axis=1)

    return pts[valid], colors[valid]

@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:

    plot = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = load_single_img(cfg)

    de = DepthEstimator(cfg, device, plot=False)
    try:
        de._load_model()
    except Exception:
        raise RuntimeError(
            "\033[33mtimm incompatibility when trying to load model. Try changing the depth model in data_config.yaml "
            "or switch to the other project environment\033[0m"
        )
    depth = de._predict(img)

    log.info(f"Raw depth stats: min={depth.min():.2f}, max={depth.max():.2f}, "
             f"mean={depth.mean():.2f}, median={depth.median():.2f}")

    T = get_camera_transform(cfg.mcap_path)

    if isinstance(img, torch.Tensor):
        _, H_orig, W_orig = img.shape
        img_size = (W_orig, H_orig)
    elif isinstance(img, np.ndarray):
        H_orig, W_orig = img.shape[:2]
        img_size = (W_orig, H_orig)
    else:
        img_size = img.size  # PIL (W, H)

    depth_np     = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)
    H_dep, W_dep = depth_np.shape

    alpha, beta  = calibrate_depth_with_radar(cfg, depth_np, img_hw=img_size[::-1], depth_hw=(H_dep, W_dep))
    depth        = torch.from_numpy(np.clip(alpha * depth_np + beta, 0, None)).to(depth.device)

    points = project_2D_3D(cfg, depth, T, img_size=img_size)
    
    img_np = np.array(img, dtype=np.float32) / 255.0
    H, W = points.shape[0], points.shape[1]
    if img_np.shape[0] != H or img_np.shape[1] != W:
        img_np = cv2.resize(img_np, (W, H))

    colors = img_np.reshape(-1, 3)
    pts = points.reshape(-1, 3).numpy()

    if plot:
        plot_lifted_points(cfg, pts, colors, img, meshlab=True)


if __name__ == '__main__':
    main()
