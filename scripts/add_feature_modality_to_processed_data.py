import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import cProfile
import numpy as np
import glob
import torchvision
import math
import cv2
from pathlib import Path

from bev_multimae.pipelines.data_pipe import BEVPipeline
from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar
from bev_multimae.preprocessing.get_transforms import apply_transform, get_all_tfs
from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.camera.camera_depth_calibration import calibrate_depth_with_sensor

import timm
import torch.nn as nn

log = logging.getLogger(__name__)


class FeatureBackbone(nn.Module):
    """ConvNeXt-Tiny stage-2 features: (B, 384, H/16, W/16)."""
    def __init__(self):
        super().__init__()
        self.net = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True, out_indices=[2]
        )

    @torch.no_grad()
    def forward(self, img_tensor):
        return self.net(img_tensor)[0]


def load_cam_info(cfg):
    cam_info = dict(np.load(cfg.camera_info))
    if np.all(cam_info['D'] == 0):
        event = Path(cfg.mcap_path).stem
        event_k1 = {
            "evt_0dyEeORn8jJHCOq2": -3.36873119e-01,
            "evt_0dz4DPX49GHES3QR": -3.16873119e-01,
            "evt_0e8QmDh0sX27pfVW": -3.36873119e-01,
            "evt_0e8QmOXkpfZKY1ih": -3.36873119e-01,
            "evt_0e8QmXqaenvFbugE": -3.36873119e-01,
            "evt_0e8Qmgb6bdukqOwj": -2.36873119e-01,
            "evt_0e8QmsFL0xUEIEWj": -2.36873119e-01,
            "evt_0e8Qn5kqXWs3r28T": -2.36873119e-01,
            "evt_0e8QnBKXiE4mtDNl": -2.76873119e-01,
            "evt_0e8QndIYxPSnB0y9": -2.76873119e-01,
        }
        k1 = event_k1[event]
        cam_info['D'] = np.array([k1, 1.29256173e-01, 1.02774231e-03, 1.23003590e-04, -2.42683235e-02])

    return cam_info['K'], cam_info['D']


def compute_grid_size(voxel_size, point_cloud_range) -> list:
    pcr = point_cloud_range
    return [
        math.ceil((pcr[3] - pcr[0]) / voxel_size[0]),
        math.ceil((pcr[4] - pcr[1]) / voxel_size[1]),
        math.ceil((pcr[5] - pcr[2]) / voxel_size[2]),
    ]


def ray_directions(K, D, img_hw, feat_hw) -> torch.Tensor:
    """
    Undistorted unit ray direction for each feature pixel.
    Returns (N, 3) where N = Hf * Wf.
    """
    Hf, Wf = feat_hw
    H_img, W_img = img_hw

    K = K.copy()
    K[0, :] *= Wf / W_img
    K[1, :] *= Hf / H_img

    u, v = np.meshgrid(
        np.arange(Wf, dtype=np.float32) + 0.5,
        np.arange(Hf, dtype=np.float32) + 0.5,
    )
    pixels = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pixels, K, D).reshape(-1, 2)  # (N, 2)
    rays = np.concatenate([norm, np.ones((len(norm), 1), dtype=np.float32)], axis=-1)
    return torch.from_numpy(rays)  # (N, 3)


def lss_splat(
    feat_map: torch.Tensor,
    depth,
    T_cam_ego: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    img_hw: tuple,
    voxel_size: list,
    point_cloud_range: list,
    grid_size: list,
    depth_bins: torch.Tensor,
    sigma_depth: float = 3.0,
    z_max: float = 5.0,
    bev_fill_kernel: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    LSS-style lift-splat into BEV.

    Each feature pixel is spread along its camera ray across D depth bins,
    weighted by a Gaussian centred on the MoGe-predicted depth.
    This replaces the hard single-depth assignment with a soft distribution,
    so depth uncertainty is reflected in the BEV rather than ignored.

    Returns (C, H_bev, W_bev).
    """
    C, Hf, Wf = feat_map.shape
    N = Hf * Wf
    n_bins = len(depth_bins)
    H_bev, W_bev = grid_size[1], grid_size[0]
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]

    depth_np = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)
    depth_np = np.clip(depth_np, 0.0, depth_bins[-1].item())  # kill inf/nan sky pixels
    if depth_np.shape != (Hf, Wf):
        depth_np = cv2.resize(depth_np, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
    depth_pred = torch.from_numpy(depth_np).reshape(N).to(device)  # (N,)

    rays = ray_directions(K, D, img_hw, (Hf, Wf)).to(device)      # (N, 3)
    depth_bins = depth_bins.to(device)                              # (D,)

    # 3D points along each ray for every depth bin: (N, D, 3)
    pts_cam = rays[:, None, :] * depth_bins[None, :, None]

    # Transform all points to ego frame at once
    T = torch.from_numpy(T_cam_ego).float().to(device)
    pts_h = torch.cat(
        [pts_cam.reshape(-1, 3), torch.ones(N * n_bins, 1, device=device)], dim=-1
    )
    pts_ego = (T @ pts_h.T).T[:, :3].reshape(N, n_bins, 3)        # (N, D, 3)

    # # Gaussian depth weights centred on predicted depth, normalised per pixel
    # weights = torch.exp(
    #     -0.5 * (depth_bins[None, :] - depth_pred[:, None]) ** 2 / sigma_depth ** 2
    # )
    # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # (N, D)
    depth_diff = torch.abs(depth_bins[None, :] - depth_pred[:, None])

    weights = torch.exp(-0.5 * depth_diff ** 2 / sigma_depth ** 2)

    surface_mask = depth_diff <= (2.0 * sigma_depth)
    weights = weights * surface_mask.float()

    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    # BEV grid indices
    bx = ((pts_ego[..., 0] - x_min) / voxel_size[0]).long()       # (N, D)
    by = ((pts_ego[..., 1] - y_min) / voxel_size[1]).long()       # (N, D)

    valid = (
        (bx >= 0) & (bx < W_bev) &
        (by >= 0) & (by < H_bev) &
        (pts_ego[..., 2] < z_max)
    )

    flat_idx = (by * W_bev + bx).clamp(0, H_bev * W_bev - 1)     # (N, D)
    feats = feat_map.permute(1, 2, 0).reshape(N, C).to(device)    # (N, C)

    bev = torch.zeros(H_bev * W_bev, C, device=device)
    bev_w = torch.zeros(H_bev * W_bev, device=device)

    # Scatter one depth bin at a time to avoid (N * D * C) memory spike
    for d in range(n_bins):
        mask = valid[:, d]
        if not mask.any():
            continue
        idx = flat_idx[:, d][mask]          # (M,)
        w   = weights[:, d][mask]           # (M,)
        f   = feats[mask]                   # (M, C)
        bev.scatter_add_(0, idx[:, None].expand(-1, C), f * w[:, None])
        bev_w.scatter_add_(0, idx, w)

    filled = bev_w > 0
    bev[filled] /= bev_w[filled, None]

    bev = bev.reshape(H_bev, W_bev, C).permute(2, 0, 1)  # (C, H_bev, W_bev)

    return bev


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    direction = cfg.direction
    if direction == 'left':
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/left")
        save_processed = "data/processed_2/left"
    elif direction == 'right':
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/right")
        save_processed = "data/processed_2/right"
    else:
        raise ValueError(f"Unknown direction: {direction!r}. Expected 'left' or 'right'.")

    log.info('Initializing pipeline...')
    pipeline = BEVPipeline(cfg)
    log.info('Pipeline initialized')

    events = sorted(os.listdir(cfg.mcap_extract_path))
    n_events = len(events)

    backbone = FeatureBackbone().eval().to(cfg.device)
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_events = [
        'evt_0dpi1HrSsDgzvaw2', 'evt_0dyJyd30SYOwBKSH', 'evt_0e8RDkd1DPoqK2oQ',
        'evt_0e8QmsFL0xUEIEWj', 'evt_0e8RCA7oPH7aLq5j', 'evt_0e3qa9akdU4BIHaF',
        'evt_0e8QyaRCAyD80b9U', 'evt_0e8Qrf9nfu208uKP', 'evt_0e8Qn5kqXWs3r28T',
        'evt_0eF19uqU42iRNlag', 'evt_0e3rMOlUlNNs8CD1', 'evt_0e8QoYoT9QEHPW21',
        'evt_0e3qe7dR7iyLDlAE', 'evt_0e8RRhN10JKbTwss', 'evt_0e8Qmgb6bdukqOwj',
    ]

    model_voxel_size  = cfg.voxel_size
    bev_feat_voxel    = cfg.get("bev_feat_voxel", cfg.hi_res_voxel)

    point_cloud_range = cfg.right_point_cloud_range if direction == 'right' else cfg.left_point_cloud_range
    grid_size         = compute_grid_size(bev_feat_voxel, point_cloud_range)

    log.info(f"BEV feature voxel size: {bev_feat_voxel}")
    log.info(f"BEV feature grid size: {grid_size}")

    # Depth bins: 1–40 m, 16 bins. Adjust range/count via cfg if needed.
    depth_bins = torch.linspace(
        getattr(cfg, 'depth_min', 1.0),
        getattr(cfg, 'depth_max', 40.0),
        getattr(cfg, 'depth_bins', 16),
    )
    sigma_depth = cfg.sigma_depth

    de = DepthEstimator(cfg, cfg.device, plot=cfg.plotting)
    de._load_model()

    prof = cProfile.Profile()
    prof.enable()

    for event_idx, event in enumerate(events):
        log.info(f'Processing event {event_idx}/{n_events}')

        split = "val" if event in val_events else "train"

        load_dir = os.path.join(cfg.processed_data_dir, split, event)
        save_dir = os.path.join(save_processed, split, event)

        if os.path.exists(save_dir):
            log.info(f'Skipping {event} — already processed')
            continue

        lidar_path = os.path.join(cfg.mcap_extract_path, event, "lidar", "front_top")
        if not os.path.exists(lidar_path) or not os.listdir(lidar_path):
            log.warning(f'Skipping {event} — empty or missing lidar folder')
            continue

        radar_path = os.path.join(cfg.mcap_extract_path, event, "radar", f"front_{direction}")
        if not os.path.exists(radar_path) or not os.listdir(radar_path):
            log.warning(f'Skipping {event} — empty or missing radar folder')
            continue

        if not os.path.exists(load_dir):
            log.warning(f'Skipping {event} — missing processed folder {load_dir}')
            continue

        saved_files = sorted(glob.glob(os.path.join(load_dir, "*.pt")))
        if not saved_files:
            log.warning(f'Skipping {event} — no .pt files in {load_dir}')
            continue

        OmegaConf.update(cfg, "camera_info",    f"data/raw/mcap_extract/{event}/camera/front_{direction}/camera_info.npz")
        OmegaConf.update(cfg, "radar_raw_path", f"data/raw/mcap_extract/{event}/radar/front_{direction}")
        OmegaConf.update(cfg, "imgs_raw_path",  f"data/raw/mcap_extract/{event}/camera/front_{direction}")
        OmegaConf.update(cfg, "lidar_raw_path", f"data/raw/mcap_extract/{event}/lidar/front_top")
        OmegaConf.update(cfg, "mcap_path",      os.path.join(cfg.bags_path, f"{event}.mcap"))

        frames = sync_frames(cfg)
        if len(frames) == 0:
            log.warning(f'Skipping {event} — sync_frames returned no frames')
            continue

        if len(frames) != len(saved_files):
            log.warning(
                f'{event} — frame/file count mismatch: '
                f'{len(frames)} frames vs {len(saved_files)} .pt files; truncating to min'
            )

        os.makedirs(save_dir, exist_ok=True)

        T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, T_lid_ego = get_all_tfs(cfg, right=(direction == 'right'))
        K, D = load_cam_info(cfg)

        for j, (frame, load_path) in enumerate(zip(frames, saved_files)):
            log.info(f'Processing frame {j}/{len(frames)}')

            img    = load_img(frame['cam'])
            lidar  = load_lidar(frame['lid'])
            img_hw = (img.height, img.width)

            img_tensor = to_tensor(img).unsqueeze(0).to(cfg.device)

            with torch.inference_mode():
                feat_map = backbone(img_tensor).squeeze(0).cpu()
                depth    = de._predict(img)

            depth_np = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)

            depth_completed = calibrate_depth_with_sensor(
                cfg,
                depth_np,
                T_lid_cam=T_lid_cam,
                T_rad_cam=T_rad_cam,
                img_hw=img_hw,
                depth_hw=depth_np.shape,
                cal_pts=lidar,
                K=K,
                D=D,
                plot=cfg.plotting,
                img=img,
            )

            bev_feat = lss_splat(
                feat_map, depth_completed, T_cam_ego, K, D, img_hw,
                bev_feat_voxel, point_cloud_range, grid_size,
                depth_bins, sigma_depth, device=cfg.device,
            )

            data = torch.load(load_path, weights_only=True)
            data["bev_feat"] = bev_feat.detach().cpu().half()
            torch.save(data, os.path.join(save_dir, f"{j:06d}.pt"))


    prof.disable()
    prof.dump_stats("profile.out")


if __name__ == '__main__':
    main()