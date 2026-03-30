import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import logging
print('Matplotlib done')

# Local
from bev_multimae.preprocessing.get_transforms import apply_transform, T_cam_to_ego
from bev_multimae.preprocessing.BEV.splat import hard_splat, patchify
from bev_multimae.preprocessing.sync import sync_frames, load_img

from bev_multimae.visualization.BEV_visualization import plot_bev_comparison, overlay_radar_on_image

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

    def __init__(self, grid_size):
        super().__init__()
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
    

@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig):
   print('Should not be run at stand alone at this point')

if __name__ == '__main__':
    main()