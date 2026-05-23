import cv2
import numpy as np
import logging
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import open3d as o3d
from sklearn.cluster import DBSCAN
import math
from scipy.optimize import linear_sum_assignment

from bev_multimae.visualization.finetuning_dataviz import plot_human_boxes
from bev_multimae.preprocessing.camera.lift import load_cam_info
from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.get_transforms import get_all_tfs
from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar, load_radar, load_seg, load_bbox
from bev_multimae.preprocessing.radar.radar_process_utils import radar_to_ego
from bev_multimae.finetuning.finetuning_utils import depth_from_lidar, depth_from_seg, bbox3d_from_depth

log = logging.getLogger(__name__)

def ok_bbox(bbox, img, seg, min_px=20, border=5):
    x1, y1, x2, y2 = bbox.astype(int)
    H, W = np.array(img).shape[:2]

    if x1 <= border or y1 <= border or x2 >= W - border or y2 >= H - border:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if np.count_nonzero(seg[y1:y2, x1:x2]) < min_px:
        return False

    return True

def human_boxes_single(cfg, frame, depth_model, K, D, transforms, event, plot=False):

    img = load_img(frame["cam"])
    lidar = load_lidar(frame["lid"])
    seg = load_seg(frame["seg"])
    bboxes = load_bbox(frame["bbox"])
    box_depth = cfg.box_depth

    T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam = transforms

    boxes, centers = [], []

    for bbox in bboxes:
        if not ok_bbox(bbox, img, seg):
            boxes.append(None)
            centers.append(None)
            continue

        z = depth_from_lidar(cfg, bbox, img, lidar, seg, T_lid_cam, K, D)

        if z is None:
            z = depth_from_seg(cfg, bbox, img, lidar, seg, depth_model, T_lid_cam, T_rad_cam, K, D)

        if z is None:
            boxes.append(None)
            centers.append(None)
            continue

        box = bbox3d_from_depth(bbox, z, K, D, T_cam_ego, box_depth)
        boxes.append(box)
        centers.append(box.mean(axis=0))

    if plot and len(bboxes) > 0:
        plot_human_boxes(cfg, event, frame, img, lidar, bboxes, boxes, centers,
                         depth_model, T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, K, D)

    return boxes

def in_bev(box, pcr):
    if box is None:
        return False

    x = box[:, 0]
    y = box[:, 1]

    return (
        (x.min() >= pcr[0]) and (x.max() <= pcr[3]) and
        (y.min() >= pcr[1]) and (y.max() <= pcr[4])
    )

def save_boxes(cfg, save_folder, event, frame_idx, boxes):
    event_dir = os.path.join(save_folder, event)
    os.makedirs(event_dir, exist_ok=True)

    if cfg.direction == "right":
        pcr = cfg.right_point_cloud_range
    else:
        pcr = cfg.left_point_cloud_range

    boxes = [box for box in boxes if in_bev(box, pcr)]

    arr = np.empty(len(boxes), dtype=object)
    for i, box in enumerate(boxes):
        arr[i] = box

    np.savez(os.path.join(event_dir, f"{frame_idx:06d}.npz"), boxes=arr)
    

def human_boxes_all(cfg, event, depth_model, save_folder, plot=None):
    frames = sync_frames(cfg, seg=True)

    T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, _ = get_all_tfs(cfg, right=(cfg.direction == "right"))
    transforms = (T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam)
    K, D = load_cam_info(cfg)
    log.info(f'\nProcessing event {event}\n')

    for frame_idx, frame in enumerate(frames):
        do_plot = (frame_idx == 25) if plot is None else plot
        boxes = human_boxes_single(cfg, frame, depth_model, K, D, transforms, event, plot=do_plot)
        save_boxes(cfg, save_folder, event, frame_idx, boxes)
        log.info(f'Finished with frame {frame_idx}')