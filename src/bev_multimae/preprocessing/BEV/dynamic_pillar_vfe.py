import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import logging
import matplotlib.pyplot as plt
import os

log = logging.getLogger(__name__)


class DynamicPillarizer:

    def __init__(self, voxel_size, grid_size, point_cloud_range):
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        self.grid_size = torch.tensor(grid_size[:2])
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)

        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

    def forward(self, points):
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) /
            self.voxel_size[[0, 1]]
        ).int()

        mask = ((points_coords >= 0) &
                (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = (
            points[:, 0].int() * self.scale_xy +
            points_coords[:, 0] * self.scale_y +
            points_coords[:, 1]
        )

        unq_coords, unq_inv, unq_cnt = torch.unique(
            merge_coords, return_inverse=True, return_counts=True
        )

        num_pillars = unq_coords.shape[0]
        points_sum = torch.zeros((num_pillars, 3), device=points_xyz.device)
        points_sum.index_add_(0, unq_inv, points_xyz)
        points_mean = points_sum / unq_cnt.unsqueeze(1)

        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (
            points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset
        )
        f_center[:, 1] = points_xyz[:, 1] - (
            points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset
        )
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        unq_coords = unq_coords.int()
        pillar_coords = torch.stack((
            unq_coords // self.scale_xy,
            (unq_coords % self.scale_xy) // self.scale_y,
            unq_coords % self.scale_y
        ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        return {
            "points": points,
            "points_xyz": points_xyz,
            "pillar_coords": pillar_coords,
            "pillar_inv": unq_inv,
            "pillar_counts": unq_cnt,
            "f_cluster": f_cluster,
            "f_center": f_center,
        }


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        out_channels = out_channels if last_layer else out_channels // 2

        self.linear = nn.Linear(in_channels, out_channels, bias=not use_norm)
        if use_norm:
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x, unq_inv):
        x = self.linear(x)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)

        num_pillars = int(unq_inv.max().item()) + 1
        x_max = torch.full((num_pillars, x.shape[1]), -1e9, device=x.device)
        x_max.scatter_reduce_(0, unq_inv.unsqueeze(1).expand(-1, x.shape[1]), x, reduce="amax")

        if self.last_layer:
            return x_max
        return torch.cat([x, x_max[unq_inv, :]], dim=1)


class PointPillarScatter(nn.Module):

    def __init__(self, grid_size, num_point_features, num_filters=(64,),
                 use_norm=True, use_absolute_xyz=True, with_distance=False):
        super().__init__()
        self.nx, self.ny = grid_size
        self.use_absolute_xyz = use_absolute_xyz
        self.with_distance = with_distance

        in_channels = num_point_features if use_absolute_xyz else num_point_features - 3
        in_channels += 6  # f_cluster + f_center
        if with_distance:
            in_channels += 1

        filter_sizes = [in_channels] + list(num_filters)
        self.pfn_layers = nn.ModuleList([
            PFNLayer(filter_sizes[i], filter_sizes[i + 1],
                     use_norm=use_norm,
                     last_layer=(i == len(filter_sizes) - 2))
            for i in range(len(filter_sizes) - 1)
        ])
        self.out_features = num_filters[-1]

    def forward(self, batch_dict):
        points      = batch_dict['points']
        f_cluster   = batch_dict['f_cluster']
        f_center    = batch_dict['f_center']
        pillar_inv    = batch_dict['pillar_inv']
        pillar_coords = batch_dict['pillar_coords']

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            features.append(torch.norm(points[:, 1:4], 2, dim=1, keepdim=True))

        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, pillar_inv)

        batch_size = pillar_coords[:, 0].max().int().item() + 1
        spatial_features = torch.zeros(
            batch_size, self.out_features, self.ny, self.nx,
            dtype=features.dtype, device=features.device
        )

        batch_idx = pillar_coords[:, 0].long()
        y = pillar_coords[:, 1].long()
        x = pillar_coords[:, 2].long()
        spatial_features[batch_idx, :, y, x] = features

        batch_dict['spatial_features'] = spatial_features
        return batch_dict

def build_bev_target(batch_dict, grid_size):
    pillar_coords = batch_dict['pillar_coords']
    pillar_counts = batch_dict['pillar_counts']
    points_xyz = batch_dict['points_xyz']
    pillar_inv = batch_dict['pillar_inv']

    nx, ny = grid_size
    B = int(pillar_coords[:, 0].max().item()) + 1

    bev = torch.zeros(B, 3, ny, nx, device=points_xyz.device)

    b = pillar_coords[:, 0].long()
    y = pillar_coords[:, 1].long()
    x = pillar_coords[:, 2].long()

    # occupancy
    bev[b, 0, y, x] = 1.0

    # density
    bev[b, 1, y, x] = torch.log1p(pillar_counts.float())

    # height (mean z)
    num_pillars = pillar_counts.shape[0]
    z_sum = torch.zeros(num_pillars, device=points_xyz.device)
    z_sum.index_add_(0, pillar_inv, points_xyz[:, 2])
    z_mean = z_sum / pillar_counts
    bev[b, 2, y, x] = z_mean

    return bev


def plot_bev_target(cfg, bev, name="bev_target"):

    save_folder = os.path.join(cfg.plot_folder, "BEV_target")
    os.makedirs(save_folder, exist_ok=True)

    bev = bev[0].detach().cpu()

    occ = bev[0].numpy()
    dens = bev[1].numpy()
    h = bev[2].numpy()

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axs[0].imshow(occ, origin='lower')
    axs[0].set_title("Occupancy")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(dens, origin='lower')
    axs[1].set_title("Density (log count)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(h, origin='lower')
    axs[2].set_title("Height (mean z)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    for ax in axs:
        ax.set_xlabel("Forward")
        ax.set_ylabel("Left")

    plt.tight_layout()

    save_path = os.path.join(save_folder, f"{name}.png")
    plt.savefig(save_path)
    plt.close()


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('Should not be run at stand alone at this point')

if __name__ == '__main__':
    main()