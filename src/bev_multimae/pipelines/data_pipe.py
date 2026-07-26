import torch
import numpy as np
import logging
import math
from PIL import Image
import os

# Local
from bev_multimae.preprocessing.radar.radar_process_utils import radar_to_ego, filter_radar, build_thresholds

from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.camera.lift import lift
from bev_multimae.preprocessing.BEV.splat import hard_splat, patchify
from bev_multimae.preprocessing.BEV.dynamic_pillar_vfe import DynamicPillarizer, PointPillarScatter, build_bev_target
from bev_multimae.preprocessing.sync import load_img, load_lidar, load_radar
from bev_multimae.preprocessing.get_transforms import get_all_tfs
from bev_multimae.preprocessing.lidar.lidar_process_utils import lidar_to_ego
from bev_multimae.visualization.BEV_visualization import plot_bev_target

from time import perf_counter


log = logging.getLogger(__name__)


class BEVPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.voxel_size = cfg.voxel_size
        
        if cfg.direction == 'right':
            self.point_cloud_range = cfg.right_point_cloud_range
        else:
            self.point_cloud_range = cfg.left_point_cloud_range

        self.hi_res_voxel = cfg.hi_res_voxel
        self.grid_size = self._compute_grid_size(self.voxel_size)
        self.hi_res_grid_size = self._compute_grid_size(self.hi_res_voxel)
        self.patch_size_pixels = int(self.voxel_size[0] / self.hi_res_voxel[0])
        self.radar_thresholds = build_thresholds(cfg)

        self.num_cam_channels = cfg.cam_channels
        self.num_rad_channels = cfg.rad_channels

        self.pillarizer = DynamicPillarizer(
            voxel_size=self.voxel_size,
            grid_size=self.grid_size,
            point_cloud_range=self.point_cloud_range
        )

        self.de = DepthEstimator(cfg, self.device, plot=cfg.plotting)
        self.de._load_model()

    def _compute_grid_size(self, voxel_size) -> list:
        pcr = self.point_cloud_range
        return [
            math.ceil((pcr[3] - pcr[0]) / voxel_size[0]),  # X (forward) → nx
            math.ceil((pcr[4] - pcr[1]) / voxel_size[1]),  # Y (lateral) → ny
            math.ceil((pcr[5] - pcr[2]) / voxel_size[2]),  # Z → nz
        ]

    def process(self, frame: dict) -> dict:
        right_bool = (self.cfg.direction == 'right')
        self.T_cam_ego, self.T_rad_ego, self.T_rad_cam, self.T_lid_cam, self.T_lid_ego = get_all_tfs(self.cfg, right=right_bool)
        
        img   = load_img(frame['cam'])
        radar = self._merge_radar(frame['rad'])
        lidar = load_lidar(frame['lid'])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = perf_counter()

        batch_dict_rad, bev_target, pts_rad_ego, pts_rad_sframe = self._process_radar(radar)
        pts_lidar, pts_lid_sframe = self._process_lidar(lidar)

        cal_pts = lidar if self.cfg.calibration == 'lidar' else radar
        cam_patches, bev_cam_splatted, bev_cam_hires, pts_cam_ego = self._process_img(img, cal_pts)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = perf_counter() - t0
        log.info(f"[PROFILE] BEV sample processing time, excluding loading/model init/saving: {dt:.4f} s")

        return {
            "batch_dict_rad":    batch_dict_rad,    
            "bev_radar_target":  bev_target,
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
        pts_cam_ego, colors  = lift(
                self.cfg, 
                img, 
                cal_pts, 
                self.de, 
                self.T_cam_ego, 
                self.T_lid_cam, 
                self.T_rad_cam, 
                plot=self.cfg.meshlab_cam
            )

        bev_cam_splatted = hard_splat(pts_cam_ego, colors, self.voxel_size, self.point_cloud_range, self.grid_size)
        bev_cam_hires    = hard_splat(pts_cam_ego, colors, self.hi_res_voxel, self.point_cloud_range, self.hi_res_grid_size)
        cam_patches      = patchify(bev_cam_hires, self.patch_size_pixels)
        return cam_patches, bev_cam_splatted, bev_cam_hires, pts_cam_ego
    
    def _process_radar(self, radar):
        radar = filter_radar(radar, self.radar_thresholds)
        ego_radar = radar_to_ego(self.cfg, radar, self.T_rad_ego)

        points_xyz = torch.from_numpy(ego_radar).float()
        batch_idx_rad = torch.zeros(points_xyz.shape[0], 1)
        points = torch.cat([batch_idx_rad, points_xyz], dim=1)

        batch_dict_rad = self.pillarizer.forward(points)

        bev_target = build_bev_target(batch_dict_rad, grid_size=self.grid_size[:2], num_rad_channels=self.num_rad_channels)
        if self.cfg.plotting:
            plot_bev_target(self.cfg, bev_target, name="radar_bev")

        
        return batch_dict_rad, bev_target, ego_radar, radar

    def _process_lidar(self, lidar):
        ego_lidar = lidar_to_ego(self.cfg, lidar, self.T_lid_ego)
        return ego_lidar, lidar

