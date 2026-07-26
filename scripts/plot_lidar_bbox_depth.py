import os
import cv2
import hydra
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydra.utils import to_absolute_path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import OmegaConf

from bev_multimae.preprocessing.camera.lift import load_cam_info
from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.get_transforms import apply_transform, get_all_tfs
from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar, load_bbox
from bev_multimae.finetuning.finetuning_utils import depth_from_lidar, bbox3d_from_depth
from bev_multimae.visualization.finetuning_dataviz import plot_human_boxes
from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar, load_bbox, load_seg
from bev_multimae.finetuning.finetuning_utils import depth_from_lidar, depth_from_seg, bbox3d_from_depth


EVENT = "evt_0e8RSwkcSts5kEaF"
DIRECTION = "right"
FRAME_IDX = 0
BBOX_IDX = 0
CLOSEST_PCT = 30
SAVE_DIR = "reports/figures/lidar_bbox_depth"


def cfgv(cfg, name, default):
    return cfg[name] if name in cfg else default


def set_paths(cfg, event, direction):
    root = to_absolute_path(cfg.mcap_extract_path)

    OmegaConf.update(cfg, "camera_info", os.path.join(root, event, "camera", f"front_{direction}", "camera_info.npz"), force_add=True)
    OmegaConf.update(cfg, "radar_raw_path", os.path.join(root, event, "radar", f"front_{direction}"), force_add=True)
    OmegaConf.update(cfg, "imgs_raw_path", os.path.join(root, event, "camera", f"front_{direction}"), force_add=True)
    OmegaConf.update(cfg, "seg_raw_path", os.path.join(root, event, "seg", f"front_{direction}"), force_add=True)
    OmegaConf.update(cfg, "bbox_raw_path", os.path.join(root, event, "bbox", f"front_{direction}"), force_add=True)
    OmegaConf.update(cfg, "lidar_raw_path", os.path.join(root, event, "lidar", "front_top"), force_add=True)
    OmegaConf.update(cfg, "mcap_path", to_absolute_path(f"data/raw/bags/{event}.mcap"), force_add=True)
    OmegaConf.update(cfg, "direction", direction, force_add=True)


def valid_event(cfg, event, direction):
    root = to_absolute_path(cfg.mcap_extract_path)

    paths = [
        os.path.join(root, event, "camera", f"front_{direction}", "camera_info.npz"),
        os.path.join(root, event, "camera", f"front_{direction}"),
        os.path.join(root, event, "bbox", f"front_{direction}"),
        os.path.join(root, event, "lidar", "front_top"),
    ]

    return all(os.path.exists(p) for p in paths)


def proj_lidar(img, lidar, bbox, T_lid_cam, K, D):
    x1, y1, x2, y2 = np.asarray(bbox).astype(int)
    H, W = np.asarray(img).shape[:2]

    pts = apply_transform(T_lid_cam, lidar)
    z = pts[:, 2]

    valid = np.isfinite(pts).all(axis=1) & (z > 0) & (np.abs(pts[:, 0] / z) < 1.4)
    pts = pts[valid]

    if D is None or len(np.asarray(D).reshape(-1)) == 0:
        D = np.zeros(5)

    uv, _ = cv2.projectPoints(
        pts.astype(np.float64),
        np.zeros(3),
        np.zeros(3),
        K.astype(np.float64),
        np.asarray(D).astype(np.float64),
    )

    uv = uv.reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]

    inside_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts, u, v = pts[inside_img], u[inside_img], v[inside_img]

    inside_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return pts[inside_box], u[inside_box], v[inside_box]


def plot_depth(img, bbox, pts, u, v, closest_pct, save_path):
    x1, y1, x2, y2 = np.asarray(bbox).astype(int)

    depth = pts[:, 2]
    n = max(1, int(np.ceil(len(depth) * closest_pct / 100)))

    close_idx = np.argsort(depth)[:n]
    close_depth = depth[close_idx]
    z = float(np.median(close_depth))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].imshow(img)

    sc = axes[0].scatter(
        u,
        v,
        c=depth,
        cmap="plasma_r",
        s=5,
        linewidths=0,
        alpha=0.85,
    )

    axes[0].add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
    )

    axes[0].set_xlabel("u (px)", fontsize=26)
    axes[0].set_ylabel("v (px)", fontsize=26)
    axes[0].tick_params(axis="both", labelsize=26)

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.35)

    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label("LiDAR depth [m]", fontsize=26)
    cbar.ax.tick_params(labelsize=26)

    axes[1].hist(depth, bins=10, alpha=0.6)
    axes[1].hist(close_depth, bins=30, alpha=0.8)
    axes[1].axvline(
        z,
        color="red",
        linestyle="--",
        linewidth=3,
        label=f"Selected depth = {z:.2f} m",
    )

    axes[1].set_xlabel("Depth [m]", fontsize=26)
    axes[1].set_ylabel("Count", fontsize=26)
    axes[1].tick_params(axis="both", labelsize=26)
    axes[1].legend(fontsize=18)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_boxes(cfg, bboxes, img, lidar, seg, depth_model, K, D, T_lid_cam, T_rad_cam, T_cam_ego):
    boxes, centers = [], []

    for bbox in bboxes:
        z = depth_from_lidar(cfg, bbox, img, lidar, seg, T_lid_cam, K, D)

        if z is None:
            z = depth_from_seg(
                cfg,
                bbox,
                img,
                lidar,
                seg,
                depth_model,
                T_lid_cam,
                T_rad_cam,
                K,
                D,
            )

        if z is None:
            boxes.append(None)
            centers.append(None)
            continue

        box = bbox3d_from_depth(bbox, z, K, D, T_cam_ego, cfg.box_depth)
        boxes.append(box)
        centers.append(box.mean(axis=0))

    return boxes, centers


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    direction = cfgv(cfg, "plot_direction", DIRECTION)
    event = cfgv(cfg, "plot_event", EVENT)
    frame_idx = cfgv(cfg, "plot_frame_idx", FRAME_IDX)
    bbox_idx = cfgv(cfg, "plot_bbox_idx", BBOX_IDX)
    closest_pct = cfgv(cfg, "closest_pct", CLOSEST_PCT)

    de = DepthEstimator(cfg, device='cuda', plot=False)
    de._load_model()

    root = to_absolute_path(cfg.mcap_extract_path)
    events = [event] if event is not None else sorted(os.listdir(root))

    for event in events:
        if not valid_event(cfg, event, direction):
            continue

        set_paths(cfg, event, direction)

        try:
            frames = sync_frames(cfg, seg=True)
            T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, _ = get_all_tfs(cfg, right=(direction == "right"))
            K, D = load_cam_info(cfg)
        except Exception as e:
            print(f"Skipping {event}: {e}")
            continue

        frame_ids = [frame_idx] if frame_idx is not None else range(len(frames))

        for fi in frame_ids:
            frame = frames[int(fi)]

            img = load_img(frame["cam"])
            lidar = load_lidar(frame["lid"])
            seg = load_seg(frame["seg"])
            bboxes = load_bbox(frame["bbox"])

            if len(bboxes) == 0:
                continue

            bbox_ids = [bbox_idx] if bbox_idx is not None else range(len(bboxes))

            for bi in bbox_ids:
                if int(bi) >= len(bboxes):
                    continue

                bbox = bboxes[int(bi)]
                pts, u, v = proj_lidar(img, lidar, bbox, T_lid_cam, K, D)

                save_path = to_absolute_path(
                    os.path.join(
                        SAVE_DIR,
                        f"{event}_{direction}_frame_{int(fi):06d}_bbox_{int(bi)}.png",
                    )
                )

                if len(pts) > 0:
                    plot_depth(img, bbox, pts, u, v, closest_pct, save_path)
                    print(f"Saved plot: {save_path}")
                    print(f"LiDAR points inside bbox: {len(pts)}")
                else:
                    print("No valid LiDAR points inside bbox, using segmentation fallback")

                print(f"Event: {event}")
                print(f"Frame: {int(fi)}")
                print(f"BBox: {int(bi)}")

                boxes, centers = build_boxes(
                    cfg,
                    bboxes,
                    img,
                    lidar,
                    seg,
                    de,
                    K,
                    D,
                    T_lid_cam,
                    T_rad_cam,
                    T_cam_ego,
                )

                OmegaConf.update(cfg, "finetuning_vis", to_absolute_path(SAVE_DIR), force_add=True)
                plot_human_boxes(
                    cfg, event, frame, img, lidar, bboxes, boxes, centers,
                    de, T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, K, D,
                )

                return

    print("No valid bbox with LiDAR points found.")


if __name__ == "__main__":
    main()