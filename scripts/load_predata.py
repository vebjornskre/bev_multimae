import os
from pathlib import Path

import torch
import hydra

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset

from bev_multimae.datasets.data_with_feat import BEVDataset
from bev_multimae.engines.train_utils import compute_img_stats, compute_radar_stats


def load_stats(cfg):
    stats_path = os.path.join(cfg.processed_data_dir, "mean_std.pt")

    try:
        ms = torch.load(stats_path, map_location="cpu")
        return ms["img_mean"], ms["img_std"], ms["rad_mean"], ms["rad_std"]
    except Exception:
        data_paths = [cfg.processed_data_dir_right, cfg.processed_data_dir_left]
        img_mean, img_std = compute_img_stats(data_paths)
        rad_mean, rad_std = compute_radar_stats(data_paths)

        ms = {
            "img_mean": img_mean,
            "img_std": img_std,
            "rad_mean": rad_mean,
            "rad_std": rad_std,
        }
        torch.save(ms, stats_path)

        return img_mean, img_std, rad_mean, rad_std


def make_ds(cfg, direction, split, img_mean, img_std, rad_mean, rad_std):
    if direction == "right":
        data_path = cfg.processed_data_dir_right
        point_cloud_range = cfg.right_point_cloud_range
    else:
        data_path = cfg.processed_data_dir_left
        point_cloud_range = cfg.left_point_cloud_range

    return BEVDataset(
        data_path,
        split=split,
        img_mean=img_mean,
        img_std=img_std,
        rad_mean=rad_mean,
        rad_std=rad_std,
        augment=False,
        point_cloud_range=point_cloud_range,
    )


def has_radar(sample):
    radar = sample.get("radar", {})
    points = radar.get("points", None)
    return points is not None and points.shape[0] > 0


def has_feat(sample):
    return "bev_feat" in sample and sample["bev_feat"] is not None


def feat_shape(sample):
    if not has_feat(sample):
        return None
    return tuple(sample["bev_feat"].shape)


def count_ds(ds):
    total = len(ds)
    radar_count = 0
    feat_count = 0
    radar_points = 0
    feat_shapes = set()

    for i in range(total):
        sample = ds[i]

        if has_radar(sample):
            radar_count += 1
            radar_points += sample["radar"]["points"].shape[0]

        shape = feat_shape(sample)
        if shape is not None:
            feat_count += 1
            feat_shapes.add(shape)

    radar_pct = 100.0 * radar_count / total if total > 0 else 0.0
    feat_pct = 100.0 * feat_count / total if total > 0 else 0.0
    avg_radar_points = radar_points / total if total > 0 else 0.0

    feat_shapes = sorted(feat_shapes)
    main_shape = feat_shapes[0] if len(feat_shapes) == 1 else None

    return {
        "total": total,
        "radar_count": radar_count,
        "radar_pct": radar_pct,
        "feat_count": feat_count,
        "feat_pct": feat_pct,
        "avg_radar_points": avg_radar_points,
        "feat_shapes": feat_shapes,
        "feat_shape": main_shape,
    }


def shape_text(shape):
    if shape is None:
        return "mixed / missing"
    return " x ".join(str(v) for v in shape)


def print_latex(rows):
    print("\nLaTeX table:\n")
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(r"    \caption{Overview of the pre-training dataset.}")
    print(r"    \label{tab:pretraining_dataset}")
    print(r"    \begin{tabular}{lcccccc}")
    print(r"        \toprule")
    print(r"        Split & Total samples & Samples with radar & Radar samples [\%] & Samples with BEV features & BEV feature samples [\%] & BEV feature shape \\")
    print(r"        \midrule")

    for split, stats in rows:
        print(
            f"        {split.capitalize()} & "
            f"{stats['total']} & "
            f"{stats['radar_count']} & "
            f"{stats['radar_pct']:.1f} & "
            f"{stats['feat_count']} & "
            f"{stats['feat_pct']:.1f} & "
            f"{shape_text(stats['feat_shape'])} \\\\"
        )

    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    img_mean, img_std, rad_mean, rad_std = load_stats(cfg)

    rows = []

    for split in ["train", "val"]:
        right_ds = make_ds(cfg, "right", split, img_mean, img_std, rad_mean, rad_std)
        left_ds = make_ds(cfg, "left", split, img_mean, img_std, rad_mean, rad_std)

        ds = ConcatDataset([right_ds, left_ds])
        stats = count_ds(ds)
        rows.append((split, stats))

    print("\nDataset statistics:\n")

    for split, stats in rows:
        shape = stats["feat_shape"]
        channels = shape[0] if shape is not None and len(shape) == 3 else "unknown"
        spatial = shape[1:] if shape is not None and len(shape) == 3 else "unknown"

        print(
            f"{split:5s} | "
            f"total={stats['total']:5d} | "
            f"with_radar={stats['radar_count']:5d} | "
            f"radar_pct={stats['radar_pct']:5.1f}% | "
            f"with_bev_feat={stats['feat_count']:5d} | "
            f"feat_pct={stats['feat_pct']:5.1f}% | "
            f"avg_radar_points={stats['avg_radar_points']:6.1f} | "
            f"bev_feat_shape={shape_text(shape)} | "
            f"channels={channels} | "
            f"spatial={spatial}"
        )

        if len(stats["feat_shapes"]) > 1:
            print(f"      unique bev_feat shapes: {stats['feat_shapes']}")

    print_latex(rows)


if __name__ == "__main__":
    main()