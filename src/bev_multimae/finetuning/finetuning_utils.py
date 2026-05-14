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

def make_sphere_pts(centers, radius=0.5, n=1000, color=[0, 0, 1]):
    print(centers)
    centers = np.atleast_2d(centers)
    pcd = o3d.geometry.PointCloud()
    for center in centers:
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radius + center
        colors = np.tile(color, (n, 1)).astype(np.float64)
        sphere = o3d.geometry.PointCloud()
        sphere.points = o3d.utility.Vector3dVector(pts)
        sphere.colors = o3d.utility.Vector3dVector(colors)
        pcd += sphere
    return pcd

def get_human_center(cfg, bbox, img, lid, T_lid_cam, T_cam_ego, closest_pct=30):
    x1, y1, x2, y2 = bbox.astype(int)

    cam = np.load(cfg.camera_info)
    H, W = np.array(img).shape[:2]

    pts = apply_transform(T_lid_cam, lid)
    z = pts[:, 2]

    valid = np.isfinite(pts).all(1) & (z > 0) & (np.abs(pts[:, 0] / z) < 1.4)
    pts = pts[valid]

    uv, _ = cv2.projectPoints(
        pts.astype(np.float64), np.zeros(3), np.zeros(3),
        cam["K"].astype(np.float64), cam["D"]
    )

    uv = uv.reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]

    m = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    pts, u, v = pts[m], u[m], v[m]

    m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    pts = pts[m]

    if len(pts) == 0:
        return None

    n = max(1, int(np.ceil(len(pts) * closest_pct / 100)))
    pts = pts[np.argsort(pts[:, 2])[:n]]

    z_median = np.median(pts[:, 2])

    # back-project bbox center pixel with median depth to get 3D camera-frame point
    K = cam["K"]
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    u_center = (x1 + x2) / 2
    v_center = (y1 + y2) / 2
    x_3d = (u_center - cx) * z_median / fx
    y_3d = (v_center - cy) * z_median / fy

    center_point = np.array([[x_3d, y_3d, z_median]])

    return apply_transform(T_cam_ego, center_point)[0]