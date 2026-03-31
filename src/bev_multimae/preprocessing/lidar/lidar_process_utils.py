import numpy as np
import os
import logging
import glob

from omegaconf import DictConfig

from bev_multimae.preprocessing.mcap_reader import apply_transform
from bev_multimae.preprocessing.get_transforms import T_lid_to_ego

log = logging.getLogger(__name__)

def lidar_to_ego(cfg, lidar: np.ndarray) -> np.ndarray:
    _T_lid_to_ego = T_lid_to_ego(cfg.mcap_path)
    return apply_transform(_T_lid_to_ego, lidar)

def load_and_process_lidar(cfg: DictConfig) -> np.ndarray:
    lidar_frame = cfg.lidar_frame
    # load binary pointcloud
    bin_files = sorted(glob.glob(os.path.join(cfg.lidar_raw_path, "*.bin")))
    pts = np.frombuffer(open(bin_files[lidar_frame], 'rb').read(), dtype=np.float32).reshape(-1, 3)
    xyz = pts[:, :3]

    # remove vehicle body and ground clutter
    valid = (
        (xyz[:, 0] > 3.0) &   # further in front
        (xyz[:, 2] > -3.0) &  # not below ground
        (xyz[:, 2] < 4.0)     # not above sensor height
    )

    return xyz[valid]

