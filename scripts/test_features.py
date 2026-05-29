import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
import hydra
import cv2
import glob
import os
import logging

from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.get_transforms import get_all_tfs

from add_feature_modality_to_processed_data import (
    FeatureBackbone,
    load_cam_info,
    compute_grid_size,
    lss_splat,
)

log = logging.getLogger(__name__)

def plot_bev_occupancy(bev_feat, save_path="bev_feature_occupancy.png"):
    occ = (bev_feat.detach().cpu().float().abs().sum(dim=0) > 0).float()

    plt.figure(figsize=(7, 7))
    plt.imshow(occ, origin="lower", cmap="gray")
    plt.title("BEV occupied feature cells")
    plt.colorbar(label="occupied")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def pca_to_rgb(feat: torch.Tensor, mask_empty=True, boost=1.4) -> np.ndarray:
    C, H, W = feat.shape
    flat = feat.detach().cpu().float().permute(1, 2, 0).reshape(-1, C).numpy()

    valid = np.isfinite(flat).all(axis=1)

    if mask_empty:
        valid &= np.linalg.norm(flat, axis=1) > 1e-6

    rgb = np.zeros((flat.shape[0], 3), dtype=np.float32)

    if valid.sum() < 4:
        return rgb.reshape(H, W, 3)

    x = PCA(n_components=3).fit_transform(flat[valid])

    # Normalize each PCA component separately
    lo = np.percentile(x, 1, axis=0)
    hi = np.percentile(x, 99, axis=0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

    # Increase contrast/saturation slightly
    x = (x - 0.5) * boost + 0.5
    x = np.clip(x, 0, 1)

    rgb[valid] = x
    return rgb.reshape(H, W, 3)

def downsample_depth_to_feat(depth, feat_map):
    d = depth.squeeze()
    if isinstance(d, torch.Tensor):
        d = d.detach().cpu().numpy()

    Hf, Wf = feat_map.shape[-2:]
    return cv2.resize(d, (Wf, Hf), interpolation=cv2.INTER_LINEAR)


def norm_depth(depth) -> np.ndarray:
    d = depth.squeeze()
    if isinstance(d, torch.Tensor):
        d = d.detach().cpu().numpy()
    lo, hi = np.percentile(d[np.isfinite(d)], [2, 98])
    return np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    direction = cfg.direction

    EVENT     = "evt_0e8RBt87W597dqCZ"

    ALL_IMGS = sorted(glob.glob(os.path.join(f"data/processed/{direction}/train/{EVENT}/imgs/", "*.jpg")))
    IMG_PATH  = ALL_IMGS[0]
    DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

    OmegaConf.update(cfg, "camera_info", f"data/raw/mcap_extract/{EVENT}/camera/front_{direction}/camera_info.npz")
    OmegaConf.update(cfg, "mcap_path",   f"data/raw/bags/{EVENT}.mcap")

    model_voxel_size  = cfg.voxel_size
    bev_feat_voxel    = cfg.get("bev_feat_voxel", cfg.hi_res_voxel)

    point_cloud_range = cfg.right_point_cloud_range if direction == 'right' else cfg.left_point_cloud_range
    grid_size         = compute_grid_size(bev_feat_voxel, point_cloud_range)

    log.info(f"BEV feature voxel size: {bev_feat_voxel}")
    log.info(f"BEV feature grid size: {grid_size}")

    depth_bins        = torch.linspace(
        getattr(cfg, 'depth_min', 1.0),
        getattr(cfg, 'depth_max', 40.0),
        getattr(cfg, 'depth_bins', 64),
    )
    sigma_depth = cfg.sigma_depth

    img    = Image.open(IMG_PATH).convert("RGB")
    img_hw = (img.height, img.width)

    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    backbone   = FeatureBackbone().eval().to(DEVICE)
    img_tensor = to_tensor(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        feat_map = backbone(img_tensor).squeeze(0).cpu()
    print(f"Feature map: {feat_map.shape}")

    K, D = load_cam_info(cfg)

    de = DepthEstimator(cfg, DEVICE, plot=False)
    de._load_model()
    depth = de._predict(img)

    T_cam_ego, *_ = get_all_tfs(cfg, right=(direction == 'right'))

    d = depth.squeeze()
    if isinstance(d, torch.Tensor):
        d = d.cpu().numpy()
    print(f"Depth stats — min: {d.min():.2f}  max: {d.max():.2f}  median: {np.median(d):.2f}")

    bev_feat = lss_splat(
        feat_map, depth, T_cam_ego, K, D, img_hw,
        bev_feat_voxel, point_cloud_range, grid_size,
        depth_bins, sigma_depth, device=cfg.device,
    )
    print(f"BEV feature: {bev_feat.shape}")

    plot_bev_occupancy(bev_feat, "bev_feature_occupancy.png")

    depth_down = downsample_depth_to_feat(depth, feat_map)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Input image")

    axes[0, 1].imshow(norm_depth(depth_down), cmap="magma")
    axes[0, 1].set_title(f"Downsampled depth map ({depth_down.shape[0]}x{depth_down.shape[1]})")

    axes[1, 0].imshow(pca_to_rgb(feat_map, mask_empty=False, boost=1.6))
    axes[1, 0].set_title("Feature map (PCA)")

    axes[1, 1].imshow(pca_to_rgb(bev_feat.cpu(), mask_empty=True, boost=1.8), origin="lower")
    axes[1, 1].set_title("BEV features (PCA) — LSS splat")


    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plot_folder, "feature_check.png"), dpi=150)
    plt.show()
    print("Saved feature_check.png")


if __name__ == "__main__":
    main()