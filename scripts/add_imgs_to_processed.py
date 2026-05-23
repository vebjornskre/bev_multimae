import logging
import os
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from bev_multimae.preprocessing.camera.lift import load_cam_info
from bev_multimae.preprocessing.get_transforms import get_all_tfs
from bev_multimae.preprocessing.sync import load_img, sync_frames

log = logging.getLogger(__name__)


def has_full_cam_info(path):
    if not os.path.exists(path):
        return False
    try:
        data = np.load(path)
        return all(k in data for k in ["K", "D", "T_cam_ego"])
    except Exception:
        return False


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    direction = cfg.direction
    split = cfg.split

    if direction == "left":
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/left")

    if direction == "right":
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/right")

    events = sorted(os.listdir(cfg.mcap_extract_path))
    n_events = len(events)

    for event_idx, event in enumerate(events):
        log.info(f"Processing event {event_idx}/{n_events}: {event}")

        event_raw = os.path.join(cfg.mcap_extract_path, event)
        save_dir = os.path.join(cfg.processed_data_dir, split, event)
        img_save_dir = os.path.join(save_dir, "imgs")
        cam_info_save = os.path.join(save_dir, "camera_info.npz")

        imgs_done = os.path.exists(img_save_dir) and os.listdir(img_save_dir)
        cam_info_done = has_full_cam_info(cam_info_save)

        if imgs_done and cam_info_done:
            log.info(f"Skipping {event} — imgs, K, D and T_cam_ego already saved")
            continue

        OmegaConf.update(cfg, "camera_info", os.path.join(event_raw, "camera", f"front_{direction}", "camera_info.npz"))
        OmegaConf.update(cfg, "radar_raw_path", os.path.join(event_raw, "radar", f"front_{direction}"))
        OmegaConf.update(cfg, "imgs_raw_path", os.path.join(event_raw, "camera", f"front_{direction}"))
        OmegaConf.update(cfg, "lidar_raw_path", os.path.join(event_raw, "lidar", "front_top"))
        OmegaConf.update(cfg, "mcap_path", os.path.join(cfg.bags_path, f"{event}.mcap"))

        lidar_path = os.path.join(event_raw, "lidar", "front_top")
        radar_path = os.path.join(event_raw, "radar", f"front_{direction}")
        img_path = os.path.join(event_raw, "camera", f"front_{direction}")

        if not os.path.exists(save_dir):
            log.warning(f"Skipping {event} — missing processed event folder")
            continue

        if not os.path.exists(img_path) or not os.listdir(img_path):
            log.warning(f"Skipping {event} — empty or missing image folder")
            continue

        if not os.path.exists(lidar_path) or not os.listdir(lidar_path):
            log.warning(f"Skipping {event} — empty or missing lidar folder")
            continue

        if not os.path.exists(radar_path) or not os.listdir(radar_path):
            log.warning(f"Skipping {event} — empty or missing radar folder")
            continue

        os.makedirs(img_save_dir, exist_ok=True)

        if not cam_info_done:
            K, D = load_cam_info(cfg)
            right_bool = direction == "right"
            T_cam_ego, _, _, _, _ = get_all_tfs(cfg, right=right_bool)
            np.savez(cam_info_save, K=K, D=D, T_cam_ego=T_cam_ego)

        if imgs_done:
            log.info(f"Images already saved for {event}, only updated camera info")
            continue

        frames = sync_frames(cfg)

        pt_files = sorted(
            Path(save_dir).glob("*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        if len(pt_files) < len(frames):
            log.warning(f"{event}: fewer .pt files than synced frames ({len(pt_files)} < {len(frames)})")

        for j, frame in enumerate(frames):
            if j >= len(pt_files):
                break

            log.info(f"Saving image {j}/{len(frames)}")

            img = np.array(load_img(frame["cam"]))
            save_path = os.path.join(img_save_dir, f"{pt_files[j].stem}.jpg")

            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main()