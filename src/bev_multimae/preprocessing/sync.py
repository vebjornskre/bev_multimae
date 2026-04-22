import glob
import os
import numpy as np
from pathlib import Path
from PIL import Image

def load_img(path: str = None) -> dict:
    return Image.open(path).convert("RGB")

def load_radar(path: str = None) -> dict:
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

def load_lidar(path: str = None) -> dict:
    pts = np.frombuffer(open(path, 'rb').read(), dtype=np.float32).reshape(-1, 3)
    xyz = pts[:, :3]
    
    
    valid = (
        (xyz[:, 0] > 3.0) &   # further in front
        (xyz[:, 2] > -0.5) &  # not below ground
        (xyz[:, 2] < 4.0)     # not above sensor height
    )

    return xyz[valid]

def get_files(folder, ext):
    return sorted(glob.glob(os.path.join(folder, ext)))

def find_closest_idx(target_ts: int, timestamps: np.ndarray) -> int:
    idx = np.searchsorted(timestamps, target_ts)
    # check both neighbors and pick closer one
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx] - target_ts) < abs(timestamps[idx-1] - target_ts):
        return idx
    return idx - 1

def find_n_closest_idx(target_ts, timestamps, files, n=3):
    diffs = np.abs(timestamps - target_ts)
    idxs = np.argsort(diffs)[:n]
    idxs = np.sort(idxs)
    return [files[i] for i in idxs]

def sync_frames(cfg) -> list[dict]:
    cam_files = get_files(cfg.imgs_raw_path, "*.jpg")
    rad_files = get_files(cfg.radar_raw_path, "*.bin")
    lid_files = get_files(cfg.lidar_raw_path, "*.bin")

    rad_ts = np.array([int(Path(f).stem) for f in rad_files])
    lid_ts = np.array([int(Path(f).stem) for f in lid_files])

    frames = []
    for cam_path in cam_files:
        cam_ts = int(Path(cam_path).stem)
        rad_paths = find_n_closest_idx(cam_ts, rad_ts, rad_files, n=cfg.num_radar_frames)
        lid_idx = find_closest_idx(cam_ts, lid_ts)
        frames.append({
            "cam": cam_path,
            "rad": rad_paths,
            "lid": lid_files[lid_idx],
        })

    return frames

