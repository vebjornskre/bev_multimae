import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import cProfile

from bev_multimae.pipelines.data_pipe import BEVPipeline
from bev_multimae.preprocessing.sync import sync_frames, load_img
from bev_multimae.visualization.BEV_visualization import plot_bev_comparison, overlay_radar_on_image

import timm
import torch.nn as nn

log = logging.getLogger(__name__)


class FeatureBackbone(nn.Module):
    """ConvNeXt-Tiny truncated after stage 3, returns (B, C, H/16, W/16) features."""
    def __init__(self):
        super().__init__()
        net = timm.create_model("convnext_tiny", pretrained=True, features_only=True)
        # stages 0-2 give H/8 at 192ch; stage 3 gives H/16 at 384ch — good tradeoff
        self.encoder = nn.Sequential(*list(net.children())[:4])

    @torch.no_grad()
    def forward(self, img_tensor):
        return self.encoder(img_tensor)  # (1, 384, H/16, W/16)

def _merge_radar(paths: list) -> dict:
    frames = [load_radar(p) for p in paths]
    return {k: np.concatenate([f[k] for f in frames]) for k in frames[0].keys()}


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


def hard_splat(pts_ego, features, voxel_size, point_cloud_range, grid_size):
    """Average-pool N points with (N, C) features onto a (C, H, W) BEV grid."""
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    H, W = grid_size[1], grid_size[0]
    C = features.shape[1]

    px = np.floor((pts_ego[:, 0] - x_min) / voxel_size[0]).astype(int)
    py = np.floor((pts_ego[:, 1] - y_min) / voxel_size[1]).astype(int)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, features = px[valid], py[valid], features[valid]

    bev = np.zeros((H, W, C), dtype=np.float32)
    bev_count = np.zeros((H, W), dtype=np.float32)

    np.add.at(bev, (py, px), features)
    np.add.at(bev_count, (py, px), 1)

    filled = bev_count > 0
    bev[filled] /= bev_count[filled, np.newaxis]

    return torch.from_numpy(bev).permute(2, 0, 1)  # (C, H, W)


def soft_splat(pts_ego, features, voxel_size, point_cloud_range, grid_size, sigma=0.5):
    """
    Gaussian soft splat of N points with (N, C) features onto a (C, H, W) BEV grid.

    With a trusted dense depth map each pixel has a known 3D position, so we
    use a fixed isotropic Gaussian in BEV space rather than a depth-bin
    distribution (as in LSS). sigma is in the same units as voxel_size.
    """
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    H, W = grid_size[1], grid_size[0]
    C = features.shape[1]

    bev = np.zeros((H, W, C), dtype=np.float32)
    bev_weight = np.zeros((H, W), dtype=np.float32)

    px_f = (pts_ego[:, 0] - x_min) / voxel_size[0]
    py_f = (pts_ego[:, 1] - y_min) / voxel_size[1]

    # sigma in voxel units
    sigma_vox = sigma / min(voxel_size[0], voxel_size[1])
    radius = int(np.ceil(3 * sigma_vox))

    offsets = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(offsets, offsets)
    dx, dy = dx.ravel(), dy.ravel()
    kernel_w = np.exp(-0.5 * (dx**2 + dy**2) / sigma_vox**2)  # precompute weights

    for i in range(len(dx)):
        gx = np.floor(px_f + dx[i]).astype(int)
        gy = np.floor(py_f + dy[i]).astype(int)

        valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        gx, gy = gx[valid], gy[valid]

        np.add.at(bev, (gy, gx), kernel_w[i] * features[valid])
        np.add.at(bev_weight, (gy, gx), kernel_w[i])

    filled = bev_weight > 0
    bev[filled] /= bev_weight[filled, np.newaxis]

    return torch.from_numpy(bev).permute(2, 0, 1)  # (C, H, W)


def project_2D_3D(cfg, depth, T, K, D, img_size=None):
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
    norm_coords = cv2.undistortPoints(pixel_coords, K, D).reshape(H, W, 2)

    rays = np.concatenate([norm_coords, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    points_cam = rays * depth_np[..., np.newaxis]

    pts = points_cam.reshape(-1, 3)
    valid = ~np.isnan(pts).any(axis=1) & np.isfinite(pts).all(axis=1)
    pts[valid] = apply_transform(T, pts[valid])
    pts[~valid] = np.nan

    return torch.from_numpy(pts.reshape(H, W, 3))


def lift_feature_map(
    cfg,
    feat_map: torch.Tensor,
    depth: torch.Tensor,
    T_cam_ego,
    K,
    D,
    img_hw: tuple,
    z_max: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lift a (C, Hf, Wf) feature map to ego-frame 3D points.

    depth is the dense map at the original image resolution (img_hw).
    K is scaled internally to match the feature map resolution.

    Returns pts (N, 3) and features (N, C) with invalid points removed.
    """
    C, Hf, Wf = feat_map.shape
    H_img, W_img = img_hw

    depth_np = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else np.squeeze(depth)

    # Resize depth to feature map resolution
    if depth_np.shape != (Hf, Wf):
        depth_np = cv2.resize(depth_np, (Wf, Hf), interpolation=cv2.INTER_LINEAR)

    # project_2D_3D will scale K from img_hw -> feat_map size via img_size
    ego_pts = project_2D_3D(cfg, depth_np, T_cam_ego, K, D, img_size=(W_img, H_img))

    pts = ego_pts.reshape(-1, 3).numpy()                          # (Hf*Wf, 3)
    features = feat_map.permute(1, 2, 0).reshape(-1, C).numpy()  # (Hf*Wf, C)

    valid = (
        ~np.isnan(pts).any(axis=1) &
        np.isfinite(pts).all(axis=1) &
        (pts[:, 2] < z_max)
    )
    return pts[valid], features[valid]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    direction = cfg.direction
    if direction == 'left':
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/left")
        save_processed = "data/processed_2/left"
    if direction == 'right':
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/right")
        save_processed = "data/processed_2/right"

    prof = cProfile.Profile()

    log.info('Initializing pipeline...')
    pipeline = BEVPipeline(cfg)
    log.info('Pipeline initialized')

    events = sorted(os.listdir(cfg.mcap_extract_path))
    n_events = len(events)

    backbone = FeatureBackbone().eval().to(cfg.device)
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
    ])

    prof.enable()

    val_events = [
        'evt_0dpi1HrSsDgzvaw2', 'evt_0dyJyd30SYOwBKSH', 'evt_0e8RDkd1DPoqK2oQ', 
        'evt_0e8QmsFL0xUEIEWj', 'evt_0e8RCA7oPH7aLq5j', 'evt_0e3qa9akdU4BIHaF', 
        'evt_0e8QyaRCAyD80b9U', 'evt_0e8Qrf9nfu208uKP', 'evt_0e8Qn5kqXWs3r28T', 
        'evt_0eF19uqU42iRNlag', 'evt_0e3rMOlUlNNs8CD1', 'evt_0e8QoYoT9QEHPW21', 
        'evt_0e3qe7dR7iyLDlAE', 'evt_0e8RRhN10JKbTwss', 'evt_0e8Qmgb6bdukqOwj'
        ]

    for event_idx, event in enumerate(events):
        log.info(f'Processing event {event_idx}/{n_events}')

        load_dir_root = os.path.join(cfg.processed_data_dir, "train", event)
        save_dir_root = os.path.join(save_processed, "train", event)

        load_dir = os.path.join(load_dir_root, 'val' if event in val_events else 'train')
        save_dir = os.path.join(save_dir_root, 'val' if event in val_events else 'train')

        if os.path.exists(save_dir):
            log.info(f'Skipping {event} — already processed')
            continue

        os.makedirs(save_dir, exist_ok=True)

        OmegaConf.update(cfg, "camera_info",    f"data/raw/mcap_extract/{event}/camera/front_{direction}/camera_info.npz")
        OmegaConf.update(cfg, "radar_raw_path", f"data/raw/mcap_extract/{event}/radar/front_{direction}")
        OmegaConf.update(cfg, "imgs_raw_path",  f"data/raw/mcap_extract/{event}/camera/front_{direction}")
        OmegaConf.update(cfg, "lidar_raw_path", f"data/raw/mcap_extract/{event}/lidar/front_top")
        OmegaConf.update(cfg, "mcap_path",      os.path.join(cfg.bags_path, f"{event}.mcap"))

        lidar_path = os.path.join(cfg.mcap_extract_path, event, "lidar", "front_top")
        if not os.path.exists(lidar_path) or not os.listdir(lidar_path):
            log.warning(f'Skipping {event} — empty or missing lidar folder')
            continue

        radar_path = os.path.join(cfg.mcap_extract_path, event, "radar", f"front_{direction}")
        if not os.path.exists(radar_path) or not os.listdir(radar_path):
            log.warning(f'Skipping {event} — empty or missing radar folder')
            continue

        frames = sync_frames(cfg)
        saved_files = sorted(glob.glob(os.path.join(load_dir, "*.pt")))

        for j, (frame, load_path) in enumerate(zip(frames, saved_files)):
            log.info(f'Processing frame {j}/{len(frames)}')

            img = load_img(frame['cam'])
            img_hw = (img.height, img.width)
            K, D = load_cam_info(cfg)

            img_tensor = to_tensor(img).unsqueeze(0).to(cfg.device)
            feat_map = backbone(img_tensor).squeeze(0).cpu()

            depth = de._predict(img)
            pts_ego, features = lift_feature_map(
                cfg, feat_map, depth, T_cam_ego, K, D, img_hw=img_hw
            )
            bev_feat = soft_splat(pts_ego, features, voxel_size, point_cloud_range, grid_size)

            data = torch.load(load_path)
            data["bev_feat"] = bev_feat.half()

            save_path = os.path.join(save_dir, f"{j:06d}.pt")
            torch.save(data, save_path)


if __name__ == '__main__':
    main()