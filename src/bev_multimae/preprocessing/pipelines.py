import torch
import numpy as np
import logging
import math
from PIL import Image

# Local
from bev_multimae.preprocessing.radar.radar_process_utils import radar_to_ego, filter_radar, build_thresholds

from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.camera.lift import lift
from bev_multimae.preprocessing.BEV.splat import hard_splat, patchify
from bev_multimae.preprocessing.BEV.dynamic_pillar import DynamicPillarizer, PointPillarScatter
from bev_multimae.preprocessing.sync import load_img, load_lidar, load_radar
from bev_multimae.preprocessing.lidar.lidar_process_utils import lidar_to_ego


log = logging.getLogger(__name__)


class BEVPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.voxel_size = cfg.voxel_size
        self.point_cloud_range = cfg.point_cloud_range
        self.hi_res_voxel = cfg.hi_res_voxel
        self.grid_size = self._compute_grid_size(self.voxel_size)
        self.hi_res_grid_size = self._compute_grid_size(self.hi_res_voxel)
        self.patch_size_pixels = int(self.voxel_size[0] / self.hi_res_voxel[0])
        self.radar_thresholds = build_thresholds(cfg)

        self.pillarizer = DynamicPillarizer(
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range
        )

        self.scatter = PointPillarScatter(grid_size=self.grid_size[:2])

        self.de = DepthEstimator(cfg, self.device, plot=True)
        self.de._load_model()

    def _compute_grid_size(self, voxel_size) -> list:
        pcr = self.point_cloud_range
        return [
            math.ceil((pcr[3] - pcr[0]) / voxel_size[0]),
            math.ceil((pcr[4] - pcr[1]) / voxel_size[1]),
            math.ceil((pcr[5] - pcr[2]) / voxel_size[2]),
        ]

    def process(self, frame: dict) -> dict:
        img   = load_img(frame['cam'])
        radar = self._merge_radar(frame['rad'])
        lidar = load_lidar(frame['lid'])

        bev_rad, pts_rad_ego, pts_rad_sframe = self._process_radar(radar)
        pts_lidar, pts_lid_sframe = self._process_lidar(lidar)

        cal_pts = lidar if self.cfg.calibration == 'lidar' else radar
        cam_patches, bev_cam_splatted, bev_cam_hires, pts_cam_ego = self._process_img(img, cal_pts)

        return {
            "bev_radar":         bev_rad,
            "bev_cam":           cam_patches,
            "bev_cam_hires":     bev_cam_hires,
            "bev_cam_splatted":  bev_cam_splatted,
            "pts_cam_ego":       pts_cam_ego,
            "pts_rad_ego":       pts_rad_ego,
            "pts_rad_sframe":    pts_rad_sframe,
            "pts_lid_sframe":    pts_lid_sframe
        }
    
    def _merge_radar(self, paths: list) -> dict:
        frames = [load_radar(p) for p in paths]
        return {k: np.concatenate([f[k] for f in frames]) for k in frames[0].keys()}

    def _process_img(self, img: Image.Image, cal_pts, feat: bool = False):
        pts_cam_ego, colors  = lift(self.cfg, img, cal_pts, self.de)
        bev_cam_splatted = hard_splat(pts_cam_ego, colors, self.voxel_size, self.point_cloud_range, self.grid_size)
        bev_cam_hires    = hard_splat(pts_cam_ego, colors, self.hi_res_voxel, self.point_cloud_range, self.hi_res_grid_size)
        cam_patches      = patchify(bev_cam_hires, self.patch_size_pixels)
        return cam_patches, bev_cam_splatted, bev_cam_hires, pts_cam_ego
    
    def _process_radar(self, radar):
        radar = filter_radar(radar, self.radar_thresholds)
        ego_radar = radar_to_ego(self.cfg, radar)

        points_xyz = torch.from_numpy(ego_radar).float()
        batch_idx_rad = torch.zeros(points_xyz.shape[0], 1)
        points = torch.cat([batch_idx_rad, points_xyz], dim=1)

        batch_dict_rad = self.pillarizer.forward(points)
        batch_dict_rad = self.scatter(batch_dict_rad)
        bev_rad = batch_dict_rad["spatial_features"]

        return bev_rad, ego_radar, radar

    def _process_lidar(self, lidar):
        ego_lidar = lidar_to_ego(self.cfg, lidar)
        return ego_lidar, lidar