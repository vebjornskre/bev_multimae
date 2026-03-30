# radar_processing.py

import numpy as np
import os
from pathlib import Path
import logging
import glob

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from bev_multimae.preprocessing.mcap_reader import list_transforms, apply_transform
from bev_multimae.preprocessing.get_transforms import T_rad_to_ego

log = logging.getLogger(__name__)


def load_radar_bin(path):

    # Read raw float32 buffer
    raw = np.fromfile(path, dtype=np.float32)

    points = raw.reshape(-1, 20)

    radar = {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],

        "radial_distance": points[:, 4],
        "radial_velocity": points[:, 5],
        "azimuth_angle": points[:, 6],
        "elevation_angle": points[:, 7],

        "radar_cross_section": points[:, 8],
        "signal_noise_ratio": points[:, 9],

        "radial_distance_variance": points[:, 10],
        "radial_velocity_variance": points[:, 11],
        "azimuth_angle_variance": points[:, 12],
        "elevation_angle_variance": points[:, 13],

        "radial_distance_velocity_covariance": points[:, 14],
        "velocity_resolution_processing_probability": points[:, 15],
        "azimuth_angle_probability": points[:, 16],
        "elevation_angle_probability": points[:, 17],
        "measurement_status": points[:, 18],
        "idx_azimuth_ambiguity_peer": points[:, 19],
    }

    return radar


def filter_radar(radar: dict, thresholds: dict) -> dict:
    mask = np.ones_like(radar['radial_distance'], dtype=bool)

    for field, (lo, hi) in thresholds.items():
        vals = radar[field]
        if lo is not None:
            mask &= vals >= lo
        if hi is not None:
            mask &= vals <= hi

    return {k: v[mask] for k, v in radar.items()}



def m2_to_dbsm(rcs_m2):
    return 10 * np.log10(np.maximum(rcs_m2, 1e-10))


def build_thresholds(cfg):
    return {
        "radar_cross_section": (m2_to_dbsm(cfg.rcs_m2_filter), None),
        "signal_noise_ratio": (cfg.snr_min, None),
        "radial_distance": (cfg.min_dist, cfg.max_dist),
        "elevation_angle": (cfg.elevation_angle, -cfg.elevation_angle)
    }

# def radar_to_ego(cfg, radar):
#     T_radar_to_ego = get_radar_transform(cfg.mcap_path)
#     pts_radar = np.stack([radar["x"], radar["y"], radar["z"]], axis=-1)
#     pts_radar = apply_transform(T_radar_to_ego, pts_radar)
#     pts_radar[:, 1] += 2
#     return pts_radar

def radar_to_ego(cfg, radar):
    _T_rad_to_ego = T_rad_to_ego(cfg.mcap_path)

    pts_radar = np.stack([radar["x"], radar["y"], radar["z"]], axis=-1)
    ego_pts_radar = apply_transform(_T_rad_to_ego, pts_radar)
    
    return ego_pts_radar



@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:

    path = cfg.radar_raw_path
    thresholds = build_thresholds(cfg)
    T = T_rad_to_ego(cfg.mcap_path)

    print('This script does nothing on its own')

if __name__ == '__main__':
    main()