import numpy as np
import cv2
import torch

from bev_multimae.preprocessing.get_transforms import apply_transform
from bev_multimae.preprocessing.camera.camera_depth_calibration import calibrate_depth_with_sensor


def get_all_segs(seg, bboxes, H, W):
    segs = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.astype(int)
        full_mask = np.zeros((H, W), dtype=seg.dtype)
        full_mask[y1:y2, x1:x2] = seg[y1:y2, x1:x2]
        segs.append(full_mask)
    return segs

def make_flat_bools(seg_masks, H, W):
    flat_bools = []
    for seg_mask in seg_masks:
        if seg_mask.shape != (H, W):
            seg_mask = cv2.resize(seg_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        flat_seg_mask = seg_mask.reshape(-1).astype(bool)
        flat_bools.append(flat_seg_mask)
    return np.array(flat_bools)

def get_seg_point_n_colors(flat_bools, ego_cam_pts, colors, z_threshold=30):

    segs_pts = []
    segs_colors = []

    for flat_bool in flat_bools:
        seg_pts = ego_cam_pts[flat_bool]
        seg_colors = colors[flat_bool]
        threshold = np.percentile(seg_pts[:, 0], z_threshold)
        depth_mask = seg_pts[:, 0] <= threshold

        segs_pts.append(seg_pts[depth_mask])
        segs_colors.append(seg_colors[depth_mask])

    return segs_pts, segs_colors

def depth_from_lidar(cfg, bbox, img, lidar, seg, T_lid_cam, K, D, closest_pct=30, method="bbox"):
    x1, y1, x2, y2 = bbox.astype(int)
    H, W = np.array(img).shape[:2]

    pts = apply_transform(T_lid_cam, lidar)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(1) & (z > 0) & (np.abs(pts[:, 0] / z) < 1.4)
    pts = pts[valid]

    uv, _ = cv2.projectPoints(pts.astype(np.float64), np.zeros(3), np.zeros(3), K.astype(np.float64), D)
    uv = uv.reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]

    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts, u, v = pts[m], u[m], v[m]

    m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    pts, u, v = pts[m], u[m], v[m]

    if len(pts) == 0:
        return None

    if method == "seg":
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        pts = pts[seg[v, u] > 0]

        if len(pts) == 0:
            return None

        return float(np.median(pts[:, 2]))

    n = max(1, int(np.ceil(len(pts) * closest_pct / 100)))
    pts = pts[np.argsort(pts[:, 2])[:n]]

    return float(np.median(pts[:, 2]))

def depth_from_seg(cfg, bbox, img, lidar, seg, depth_model, T_lid_cam, T_rad_cam, K, D):
    x1, y1, x2, y2 = bbox.astype(int)

    img_np = np.array(img)
    H, W = img_np.shape[:2]

    raw_depth = depth_model._predict(img)
    raw_depth = raw_depth.squeeze().cpu().numpy() if isinstance(raw_depth, torch.Tensor) else np.squeeze(raw_depth)

    depth_map = calibrate_depth_with_sensor(
        cfg, raw_depth,
        T_lid_cam=T_lid_cam, T_rad_cam=T_rad_cam,
        img_hw=(H, W), depth_hw=raw_depth.shape,
        cal_pts=lidar, K=K, D=D,
        plot=False, img=img,
    )

    # resize seg mask to depth map resolution if needed
    seg_np = np.array(seg)
    if seg_np.shape[:2] != (H, W):
        seg_np = cv2.resize(seg_np, (W, H), interpolation=cv2.INTER_NEAREST)

    human_mask = seg_np[y1:y2, x1:x2] > 0
    roi = depth_map[y1:y2, x1:x2]
    valid = roi[human_mask & np.isfinite(roi) & (roi > 0)]

    return float(np.median(valid)) if len(valid) > 0 else None


def bbox3d_from_depth(bbox, z, K, D, T_cam_ego, box_depth=0.6, human_height=1.8):
    x1, y1, x2, y2 = bbox.astype(int)

    u = (x1 + x2) / 2
    v = (y1 + y2) / 2

    pts_2d = np.array(
        [[[u, v]], [[x1, v]], [[x2, v]]],
        dtype=np.float64,
    )

    xy = cv2.undistortPoints(pts_2d, K, D).reshape(-1, 2)

    pts_cam = np.stack(
        [
            xy[:, 0] * z,
            xy[:, 1] * z,
            np.full(len(xy), z),
        ],
        axis=1,
    )

    center_ego, left_ego, right_ego = apply_transform(T_cam_ego, pts_cam)

    width = abs(right_ego[1] - left_ego[1])
    if width < 0.3:
        width = 0.6

    cx, cy, cz = center_ego

    x_min = cx - box_depth / 2
    x_max = cx + box_depth / 2
    y_min = cy - width / 2
    y_max = cy + width / 2
    z_min = cz - human_height / 2
    z_max = cz + human_height / 2

    return np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=np.float64,
    )

