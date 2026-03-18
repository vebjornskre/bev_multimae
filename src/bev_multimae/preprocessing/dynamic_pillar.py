import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

# Local
from bev_multimae.preprocessing.radar.radar_process_utils import load_and_process, build_thresholds
from bev_multimae.preprocessing.mcap_reader import (
    get_radar_transform,
    get_radar_to_camera_transform,
    apply_transform,
)
from bev_multimae.preprocessing.camera.lift import load_and_lift

# Credit of this code goes to OpenPCDet
# https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/dynamic_pillar_vfe.py
# https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py


log = logging.getLogger(__name__)

class DynamicPillarizer:

    def __init__(self, voxel_size, grid_size, point_cloud_range):

        # voxel size
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        # grid + range
        self.grid_size = torch.tensor(grid_size[:2])
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)

        # pillar center offsets
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # index scaling for unique pillar ids
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]


    def forward(self, points):

        # convert (x,y) to pillar indices
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) /
            self.voxel_size[[0, 1]]
        ).int()

        # remove points outside grid
        mask = ((points_coords >= 0) &
                (points_coords < self.grid_size[[0, 1]])).all(dim=1)

        points = points[mask]
        points_coords = points_coords[mask]

        # xyz coordinates
        points_xyz = points[:, [1, 2, 3]].contiguous()

        # merge batch + pillar indices into unique id
        merge_coords = (
            points[:, 0].int() * self.scale_xy +
            points_coords[:, 0] * self.scale_y +
            points_coords[:, 1]
        )

        # group points into pillars
        unq_coords, unq_inv, unq_cnt = torch.unique(
            merge_coords,
            return_inverse=True,
            return_counts=True
        )

        # sum xyz per pillar
        points_sum = torch.zeros(
            (unq_coords.shape[0], 3),
            device=points_xyz.device,
            dtype=points_xyz.dtype
        )

        points_sum.scatter_add_(0, unq_inv.unsqueeze(1).expand(-1, 3), points_xyz)

        # mean xyz per pillar
        points_mean = points_sum / unq_cnt.unsqueeze(1)

        # cluster feature: point offset from pillar mean
        f_cluster = points_xyz - points_mean[unq_inv, :]

        # center feature: point offset from pillar center
        f_center = torch.zeros_like(points_xyz)

        f_center[:, 0] = points_xyz[:, 0] - (
            points_coords[:, 0].to(points_xyz.dtype) *
            self.voxel_x + self.x_offset
        )

        f_center[:, 1] = points_xyz[:, 1] - (
            points_coords[:, 1].to(points_xyz.dtype) *
            self.voxel_y + self.y_offset
        )

        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        # reconstruct pillar coordinates
        unq_coords = unq_coords.int()

        pillar_coords = torch.stack((
            unq_coords // self.scale_xy,
            (unq_coords % self.scale_xy) // self.scale_y,
            unq_coords % self.scale_y
        ), dim=1)

        # reorder to (batch, y, x)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        return {
            "points_xyz": points_xyz,
            "pillar_coords": pillar_coords,
            "pillar_inv": unq_inv,
            "pillar_counts": unq_cnt,
            "f_cluster": f_cluster,
            "f_center": f_center
        }


class PointPillarScatter(nn.Module):

    def __init__(self, num_bev_features, grid_size):
        super().__init__()

        self.num_bev_features = num_bev_features
        self.nx, self.ny = grid_size


    def forward(self, batch_dict):

        # point features
        points_xyz = batch_dict['points_xyz']
        f_cluster = batch_dict['f_cluster']
        f_center = batch_dict['f_center']

        # pillar grouping
        pillar_inv = batch_dict['pillar_inv']
        pillar_counts = batch_dict['pillar_counts']
        pillar_coords = batch_dict['pillar_coords']

        # concatenate point features
        point_features = torch.cat((points_xyz, f_cluster, f_center), dim=1)

        num_pillars = pillar_coords.shape[0]
        C = point_features.shape[1]

        # aggregate point features → pillar features
        pillar_features = torch.zeros(
            (num_pillars, C),
            dtype=point_features.dtype,
            device=point_features.device
        )

        pillar_features.scatter_add_(
            0,
            pillar_inv.unsqueeze(1).expand(-1, C),
            point_features
        )

        pillar_features = pillar_features / pillar_counts.unsqueeze(1)

        # batch size
        batch_size = pillar_coords[:, 0].max().int().item() + 1

        # create BEV tensor
        spatial_features = torch.zeros(
            batch_size,
            C,
            self.ny,
            self.nx,
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )

        # coordinates
        batch_idx = pillar_coords[:, 0].long()
        y = pillar_coords[:, 1].long()
        x = pillar_coords[:, 2].long()

        # scatter pillar features into BEV grid
        spatial_features[batch_idx, :, y, x] = pillar_features

        batch_dict['spatial_features'] = spatial_features

        return batch_dict
    
def splat_rgb_bev(pts_cam, colors, voxel_size, point_cloud_range, grid_size):
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    H, W = grid_size[1], grid_size[0]

    px = np.floor((pts_cam[:, 0] - x_min) / voxel_size[0]).astype(int)
    py = np.floor((pts_cam[:, 1] - y_min) / voxel_size[1]).astype(int)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, colors = px[valid], py[valid], colors[valid]

    bev_rgb = np.zeros((H, W, 3), dtype=np.float32)
    bev_count = np.zeros((H, W), dtype=np.float32)

    # average RGB per cell
    np.add.at(bev_rgb, (py, px), colors)
    np.add.at(bev_count, (py, px), 1)

    filled = bev_count > 0
    bev_rgb[filled] /= bev_count[filled, np.newaxis]

    return torch.from_numpy(bev_rgb).permute(2, 0, 1)  # (3, H, W)



def plot_bev_comparison(cfg, pts_radar, pts_cam, colors, bev_rad, bev_cam, voxel_size, point_cloud_range):
    save_folder = os.path.join(cfg.plot_folder, "BEV")
    os.makedirs(save_folder, exist_ok=True)

    bev_rad_np = bev_rad[0].cpu().numpy()
    bev_cam_np = bev_cam.permute(1, 2, 0).numpy()  # (H, W, 3)

    H, W = bev_cam_np.shape[:2]
    pcr = point_cloud_range
    x_min, y_min = pcr[0], pcr[1]
    x_max, y_max = pcr[3], pcr[4]

    x_ticks = np.linspace(0, W, 6)
    y_ticks = np.linspace(0, H, 6)
    x_labels = [f"{v:.0f}m" for v in np.linspace(x_min, x_max, 6)]
    y_labels = [f"{v:.0f}m" for v in np.linspace(y_min, y_max, 6)]

    def apply_ticks(ax):
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("x / forward (m)")
        ax.set_ylabel("y / lateral (m)")

    def draw_grid(ax, W, H, color="white"):
        v_lines = [[(x - 0.5, -0.5), (x - 0.5, H - 0.5)] for x in range(0, W + 1)]
        h_lines = [[(-0.5, y - 0.5), (W - 0.5, y - 0.5)] for y in range(0, H + 1)]
        ax.add_collection(LineCollection(v_lines + h_lines, colors=color, linewidths=0.2, alpha=0.2, zorder=2))

    # Convert points to voxel coords
    px_rad = (pts_radar[:, 0] - x_min) / voxel_size[0] -0.5
    py_rad = (pts_radar[:, 1] - y_min) / voxel_size[1] -0.5
    px_cam = np.floor((pts_cam[:, 0] - x_min) / voxel_size[0])
    py_cam = np.floor((pts_cam[:, 1] - y_min) / voxel_size[1])

    valid_cam = (px_cam >= 0) & (px_cam < W) & (py_cam >= 0) & (py_cam < H)
    px_cam, py_cam, colors_valid = px_cam[valid_cam], py_cam[valid_cam], colors[valid_cam]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Radar occupancy + points
    occ_rad = np.any(bev_rad_np != 0, axis=0)
    axes[0].imshow(occ_rad, origin="lower", cmap="gray", aspect="auto")
    draw_grid(axes[0])
    axes[0].scatter(px_rad, py_rad, s=6, c="red", alpha=0.6, zorder=3, label="Radar points")
    axes[0].set_xlim(-0.5, W - 0.5)
    axes[0].set_ylim(-0.5, H - 0.5)
    apply_ticks(axes[0])
    axes[0].set_title("Radar BEV Occupancy")
    axes[0].legend(fontsize=8)

    # Camera RGB BEV
    axes[1].imshow(bev_cam_np, origin="lower", aspect="auto")
    draw_grid(axes[1])
    axes[1].set_xlim(-0.5, W - 0.5)
    axes[1].set_ylim(-0.5, H - 0.5)
    apply_ticks(axes[1])
    axes[1].set_title("Camera RGB BEV (splatted)")

    # Occupancy overlay
    occ_cam = np.any(bev_cam_np != 0, axis=-1)
    overlay = np.zeros((H, W, 3))
    overlay[occ_cam & ~occ_rad]  = [0.2, 0.8, 0.2]  # camera only: green
    overlay[occ_rad & ~occ_cam]  = [0.8, 0.2, 0.2]  # radar only: red
    overlay[occ_cam & occ_rad]   = [1.0, 1.0, 1.0]  # both: white

    axes[2].imshow(overlay, origin="lower", aspect="auto")
    draw_grid(axes[2], color="gray")
    axes[2].set_xlim(-0.5, W - 0.5)
    axes[2].set_ylim(-0.5, H - 0.5)
    apply_ticks(axes[2])
    axes[2].set_title("Occupancy: Camera (green) / Radar (red) / Both (white)")

    plt.suptitle("Radar vs Camera BEV", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "bev_comparison.png"), dpi=150)
    plt.close(fig)


    
@hydra.main(config_path="../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig):

    voxel_size = [5, 5, 25]
    point_cloud_range = [0, -10, -5, 50, 50, 20]
    grid_size = [
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
    ]

    pillarizer = DynamicPillarizer(
        voxel_size=voxel_size,
        grid_size=grid_size,
        point_cloud_range=point_cloud_range
    )

    scatter_rad = PointPillarScatter(
        num_bev_features=9,
        grid_size=grid_size[:2]
    )

    log.info('Loading, filtering and transfroming radar frame...')
    # Radar
    thresholds = build_thresholds(cfg)
    radar = load_and_process(cfg, thresholds)
    T_radar_to_ego = get_radar_transform(cfg.mcap_path)

    pts_radar = np.stack([radar["x"], radar["y"], radar["z"]], axis=-1)
    pts_radar = apply_transform(T_radar_to_ego, pts_radar)

    points_xyz = torch.from_numpy(pts_radar).float()
    batch_idx_rad = torch.zeros(points_xyz.shape[0], 1)
    points = torch.cat([batch_idx_rad, points_xyz], dim=1)

    log.info('Radar data ready, starting pillerization...')

    batch_dict_rad = pillarizer.forward(points)
    log.info(f"num radar pillars: {batch_dict_rad['pillar_coords'].shape[0]}")

    log.info('Scattering points to pillars')
    batch_dict_rad = scatter_rad(batch_dict_rad)
    bev_rad = batch_dict_rad["spatial_features"]
    log.info(f"Radar BEV shape: {bev_rad.shape}")

    log.info('Loading and lifting camera frame')
    # Camera
    pts_cam, colors = load_and_lift(cfg)

    log.info('Splatting camera frame onto BEV grid')
    bev_cam = splat_rgb_bev(pts_cam, colors, voxel_size, point_cloud_range, grid_size)
    log.info(f"Camera BEV shape: {bev_cam.shape}")

    plot_bev_comparison(cfg, pts_radar, pts_cam, colors, bev_rad, bev_cam, voxel_size, point_cloud_range)
    
if __name__ == '__main__':
    main()