# radar_processing.py

import numpy as np
import os
from pathlib import Path
import logging
import glob

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from bev_multimae.preprocessing.mcap_reader import list_transforms, apply_transform, get_radar_transform

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


def to_base_link(radar: dict, T: np.ndarray) -> dict:
    # Transform points from radar_front_right frame to base_link
    pts = np.stack([radar["x"], radar["y"], radar["z"]], axis=-1)
    pts = apply_transform(T, pts)
    return {**radar, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}


def m2_to_dbsm(rcs_m2):
    return 10 * np.log10(np.maximum(rcs_m2, 1e-10))


def build_thresholds(cfg):
    return {
        "radar_cross_section": (m2_to_dbsm(cfg.rcs_m2_filter), None),
        "signal_noise_ratio": (cfg.snr_min, None),
        "radial_distance": (cfg.min_dist, cfg.max_dist),
        "elevation_angle": (cfg.elevation_angle, -cfg.elevation_angle)
    }


def load_and_process(cfg, thresholds, T=None):

    radar_folder = Path(to_absolute_path(cfg.radar_raw_path))
    idx = cfg.radar_frame

    ext = "*.bin"
    radar_files = glob.glob(os.path.join(radar_folder, ext))

    radar_path = sorted(radar_files)[idx]
    log.info(f"Processing radar frame: {Path(radar_path).name}")

    # Load, filter and transform to base_link
    radar = load_radar_bin(radar_path)
    radar = filter_radar(radar, thresholds)
    if T is not None:
        radar = to_base_link(radar, T)
    return radar


def compute_stats(radar_dir: str, T=None, thresholds=None, percentiles=(1, 99)):
    all_stats = {
        'radial_distance': [],
        'azimuth_angle': [],
        'elevation_angle': [],
        'radial_velocity': [],
        'radar_cross_section': [],
        'signal_noise_ratio': [],
        'radial_distance_variance': [],
        'azimuth_angle_variance': [],
        'elevation_angle_variance': [],
        'azimuth_angle_probability': [],
        'elevation_angle_probability': [],
        'velocity_resolution_processing_probability': [],
    }

    path = Path(radar_dir)
    pts_per_scan = []
    

    for fname in os.listdir(path):
        radar = load_radar_bin(str(path / fname))
        radar = to_base_link(radar, T)

        if thresholds is not None:
            radar = filter_radar(radar, thresholds)

        pts_per_scan.append(len(radar['radial_distance']))
        for k in all_stats:
            all_stats[k].append(radar[k])

    combined = {k: np.concatenate(v) for k, v in all_stats.items()}

    lb, ub = percentiles
    report = {}
    for k, vals in combined.items():
        report[k] = {
            'min': np.min(vals),
            'max': np.max(vals),
            f'p{lb}': np.percentile(vals, lb),
            f'p{ub}': np.percentile(vals, ub),
            'mean': np.mean(vals),
            'median': np.median(vals),
            'std': np.std(vals),
        }

    pts_per_scan = np.array(pts_per_scan)
    scan_stats = {
        'min': int(np.min(pts_per_scan)),
        'max': int(np.max(pts_per_scan)),
        'mean': float(np.mean(pts_per_scan)),
        'median': float(np.median(pts_per_scan)),
        'std': float(np.std(pts_per_scan)),
    }

    return report, combined, scan_stats


def print_stats(report, combined):
    print("\nPolar statistics\n")

    r = report['radial_distance']
    print(f"Radial distance (m)")
    print(f"  min {r['min']:.2f}   p1 {r['p1']:.2f}   median {r['median']:.2f}   p99 {r['p99']:.2f}   max {r['max']:.2f}\n")

    el = report['elevation_angle']
    print(f"Elevation angle (deg)")
    print(f"  min {np.degrees(el['min']):.1f}   p1 {np.degrees(el['p1']):.1f}   p99 {np.degrees(el['p99']):.1f}   max {np.degrees(el['max']):.1f}\n")

    az = report['azimuth_angle']
    print(f"Azimuth angle (deg)")
    print(f"  min {np.degrees(az['min']):.1f}   p1 {np.degrees(az['p1']):.1f}   p99 {np.degrees(az['p99']):.1f}   max {np.degrees(az['max']):.1f}\n")

    rcs = report['radar_cross_section']
    print(f"Radar cross section (dBsm)")
    print(f"  min {rcs['min']:.1f}   p1 {rcs['p1']:.1f}   p5 {np.percentile(combined['radar_cross_section'],5):.1f}   median {rcs['median']:.1f}   p99 {rcs['p99']:.1f}   max {rcs['max']:.1f}\n")

    snr = report['signal_noise_ratio']
    print(f"Signal to noise ratio (dB)")
    print(f"  min {snr['min']:.1f}   p1 {snr['p1']:.1f}   median {snr['median']:.1f}   max {snr['max']:.1f}\n")

    for k in ['radial_distance_variance', 'azimuth_angle_variance', 'elevation_angle_variance']:
        v = report[k]
        print(f"{k}")
        print(f"  p95 {np.percentile(combined[k],95):.4f}   max {v['max']:.4f}\n")

    for k in ['azimuth_angle_probability', 'elevation_angle_probability']:
        p = report[k]
        print(f"{k}")
        print(f"  min {p['min']:.2f}   p5 {np.percentile(combined[k],5):.2f}   median {p['median']:.2f}\n")


@hydra.main(config_path="../../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{cfg}")

    path = cfg.radar_raw_path
    thresholds = build_thresholds(cfg)
    T = get_radar_transform(cfg.mcap_path)

    print('\nBEFORE FILTERING')
    report, combined, scan_stats = compute_stats(path, T)
    print_stats(report, combined)

    print("Points per scan")
    print(f"  min {scan_stats['min']}   median {scan_stats['median']:.1f}   mean {scan_stats['mean']:.1f}   max {scan_stats['max']}   std {scan_stats['std']:.1f}")

    print('\nAFTER FILTERING')
    report, combined, scan_stats = compute_stats(path, T, thresholds)
    print_stats(report, combined)

    print("Points per scan")
    print(f"  min {scan_stats['min']}   median {scan_stats['median']:.1f}   mean {scan_stats['mean']:.1f}   max {scan_stats['max']}   std {scan_stats['std']:.1f}")


if __name__ == '__main__':
    main()