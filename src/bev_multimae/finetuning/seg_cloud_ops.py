import numpy as np
import cv2

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