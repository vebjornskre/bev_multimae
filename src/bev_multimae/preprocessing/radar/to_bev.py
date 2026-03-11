# Trun voxel grid into BEV

################################
# I NEED TO ANALYZE DATA DISTRIBUTION IN POLAR COORDINATES
# THEN GO TO XYZ. FOR EXAMPLE, IF I REMOVE EVERYTHING TALLER
# THAN 3 METERS I MIGHT KEEP NOISE THAT IS CLOSE AND REOMVE SIGNLAL
# FAR AWAY
################################

import logging
import os
from pathlib import Path

import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


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

def get_point_ranges(cfg: DictConfig):

    # Path to raw radar dir
    path = Path(to_absolute_path(cfg.radar_raw_path))
    lb_percent = cfg.lb_percent
    ub_percent = cfg.ub_percent

    x_ranges = []
    y_ranges = []
    z_ranges = []

    for fname in os.listdir(path):
        radar = load_radar_bin(path=str(path / fname))

        x_range = (np.percentile(radar['x'], lb_percent[0]), np.percentile(radar['x'], ub_percent[0]))
        y_range = (np.percentile(radar['y'], lb_percent[1]), np.percentile(radar['y'], ub_percent[1]))

        # All values that are very negative in the z direction 
        # are probably multipath signals, so we exclude them    
        z_min = np.min(radar['z'][radar['z'] >= -1.5]) if np.any(radar['z'] >= -1.5) else -1.5

        z_range = (z_min, np.percentile(radar['z'], ub_percent[2]))

        x_ranges.append(x_range)
        y_ranges.append(y_range)
        z_ranges.append(z_range)

    mean_x_lb = np.mean(np.array(x_ranges)[:, 0])
    mean_x_ub = np.mean(np.array(x_ranges)[:, 1])

    mean_y_lb = np.mean(np.array(y_ranges)[:, 0])
    mean_y_ub = np.mean(np.array(y_ranges)[:, 1])

    mean_z_lb = np.mean(np.array(z_ranges)[:, 0])
    mean_z_ub = np.mean(np.array(z_ranges)[:, 1])
    

    print(f'\nmean x-range: {round(mean_x_lb, 2)} m , {mean_x_ub} m')
    print(f'mean y-range: {round(mean_y_lb, 2)} m , {mean_y_ub} m')
    print(f'mean z-range: {round(mean_z_lb, 2)} m , {mean_z_ub} m')

    return (mean_x_lb, mean_x_ub), (mean_y_lb, mean_y_ub), (mean_z_lb, mean_z_ub)


def to_bev(cfg: DictConfig):
    ...


@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{cfg}")

    x_range, y_range, z_range = get_point_ranges(cfg)

    to_bev(cfg)

if __name__ == '__main__':
    main()


    