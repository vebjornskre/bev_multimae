import glob
import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch

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

def load_seg(path: str = None):
    if path is None or not os.path.exists(path):
        return None

    seg = np.load(path)  # (H, W)
    return seg
    # return torch.from_numpy(seg).float()
    # return torch.from_numpy(seg).long()

def load_bbox(path: str = None):
    if path is None or not os.path.exists(path):
        return None
    return np.load(path)

def get_files(folder, ext):
    return sorted(glob.glob(os.path.join(folder, ext)))

def has_seg_points(path) -> bool:
    return np.any(np.load(path) != 0)

def find_closest_idx(target_ts: int, timestamps: np.ndarray) -> int:
    idx = np.searchsorted(timestamps, target_ts)
    # check both neighbors and pick closer one
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx] - target_ts) < abs(timestamps[idx-1] - target_ts):
        return idx 
    # return idx
    return idx - 1

def find_n_closest_idx(target_ts, timestamps, files, n=3):
    diffs = np.abs(timestamps - target_ts)
    idxs = np.argsort(diffs)[:n]
    idxs = np.sort(idxs)
    return [files[i] for i in idxs]


def sync_frames(cfg, seg=False) -> list[dict]:
    cam_files = get_files(cfg.imgs_raw_path, "*.jpg")
    rad_files = get_files(cfg.radar_raw_path, "*.bin")
    lid_files = get_files(cfg.lidar_raw_path, "*.bin")
    seg_files = get_files(cfg.seg_raw_path, "*.npy")
    bbox_files = get_files(cfg.bbox_raw_path, "*.npy")

    print(len(cam_files))

    rad_ts = np.array([int(Path(f).stem) for f in rad_files])
    lid_ts = np.array([int(Path(f).stem) for f in lid_files])
    seg_ts = np.array([int(Path(f).stem) for f in seg_files])
    bbox_ts = np.array([int(Path(f).stem) for f in bbox_files])

    seg_map = {int(Path(f).stem): f for f in seg_files}
    bbox_map = {int(Path(f).stem): f for f in bbox_files}

    frames = []
    for cam_path in cam_files:
        cam_ts = int(Path(cam_path).stem)

        rad_paths = find_n_closest_idx(cam_ts, rad_ts, rad_files, n=cfg.num_radar_frames)
        lid_idx = find_closest_idx(cam_ts, lid_ts)
        
        seg_idx = find_closest_idx(cam_ts, seg_ts)
        bbox_idx = find_closest_idx(cam_ts, bbox_ts)
        seg_path = seg_files[seg_idx] if len(seg_files) > 0 else None
        bbox_path = bbox_files[bbox_idx] if len(bbox_files) > 0 else None

        time_delta_ms = abs(lid_ts[lid_idx] - cam_ts) / 1e6
        # print(f"cam_ts: {cam_ts}, lid_ts: {lid_ts[lid_idx]}, delta: {time_delta_ms:.1f}ms")

        # seg_path = seg_map.get(cam_ts, None)
        # bbox_path = bbox_map.get(cam_ts, None)

        if seg and (seg_path is None or not has_seg_points(seg_path)):
            continue

        frames.append({
            "cam": cam_path,
            "rad": rad_paths,
            "lid": lid_files[lid_idx],
            "seg": seg_path,
            "bbox": bbox_path
        })

    return frames

