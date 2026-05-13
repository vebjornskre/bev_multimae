import cv2
import numpy as np
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

from bev_multimae.visualization.finetuning import plot_img_and_seg, scatter_seg_points
from bev_multimae.preprocessing.camera.lift import lift
from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.get_transforms import get_all_tfs
from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar, load_radar, load_seg, load_bbox
from bev_multimae.preprocessing.radar.radar_process_utils import radar_to_ego
from bev_multimae.visualization.BEV_visualization import overlay_radar_on_image, plot_bev_comparison
from bev_multimae.preprocessing.BEV.splat import hard_splat, patchify
from bev_multimae.preprocessing.camera.camera_depth_calibration import project_points_to_image
from bev_multimae.preprocessing.get_transforms import apply_transform


def _merge_radar(paths: list) -> dict:
        frames = [load_radar(p) for p in paths]
        return {k: np.concatenate([f[k] for f in frames]) for k in frames[0].keys()}

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

def _compute_grid_size(voxel_size, point_cloud_range) -> list:
    pcr = point_cloud_range
    return [
        math.ceil((pcr[3] - pcr[0]) / voxel_size[0]),  # X (forward) → nx
        math.ceil((pcr[4] - pcr[1]) / voxel_size[1]),  # Y (lateral) → ny
        math.ceil((pcr[5] - pcr[2]) / voxel_size[2]),  # Z → nz
    ]

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

    n = max(1, int(len(pts) * closest_pct / 100))
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

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    events = sorted(os.listdir(cfg.mcap_extract_path))
    n_events = len(events)

    direction='right'
    seg_method = False
    lid_method = not seg_method

    # event = 'evt_0e8QraX8B9UIyxY9' # evening, two people to the right in the frame
    # event = 'evt_0e8RO9yx2kWoavOD'   # afternoon, person standing still in front of robot while its driving
    # event = 'evt_0e8RSwkcSts5kEaF'  # Two people further away
    # event = 'evt_0e3qa9akdU4BIHaF' 
    event = 'evt_0e8Qmgb6bdukqOwj'  # Lacks distortion coefficients 
    # event = 'evt_0e3rMglACswVRZ1U'    # At night one person
    # event = 'evt_0e8RSh4iCdd5BTkn'

    frame_idx = 0

    OmegaConf.update(cfg, "camera_info", f"data/raw/mcap_extract/{event}/camera/front_{direction}/camera_info.npz")
    OmegaConf.update(cfg, "radar_raw_path", f"data/raw/mcap_extract/{event}/radar/front_{direction}")
    OmegaConf.update(cfg, "imgs_raw_path", f"data/raw/mcap_extract/{event}/camera/front_{direction}")
    OmegaConf.update(cfg, "seg_raw_path", f"data/raw/mcap_extract/{event}/seg/front_{direction}")
    OmegaConf.update(cfg, "bbox_raw_path", f"data/raw/mcap_extract/{event}/bbox/front_{direction}")
    OmegaConf.update(cfg, "lidar_raw_path", f"data/raw/mcap_extract/{event}/lidar/front_top")
    OmegaConf.update(cfg, "mcap_path", f"data/raw/bags/{event}.mcap")


    lidar_path = os.path.join(cfg.mcap_extract_path, event, "lidar", "front_top")
    if not os.path.exists(lidar_path) or not os.listdir(lidar_path):
        print(f'Skipping {event} — empty or missing lidar folder')

    radar_path = os.path.join(cfg.mcap_extract_path, event, "radar", f"front_{direction}")
    if not os.path.exists(radar_path) or not os.listdir(radar_path):
        print(f'Skipping {event} — empty or missing radar folder')

    frames = sync_frames(cfg, seg=True)
    print(len(frames))

    frame = frames[frame_idx]
    de = DepthEstimator(cfg, device='cuda', plot=cfg.plotting)
    de._load_model()

    img   = load_img(frame['cam'])
    radar = _merge_radar(frame['rad'])
    lidar = load_lidar(frame['lid'])
    seg = load_seg(frame['seg'])
    bboxes = load_bbox(frame['bbox'])

    cal_pts = lidar

    T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, T_lid_ego = get_all_tfs(cfg, right=True)

    ego_radar = radar_to_ego(cfg, radar, T_rad_ego)
    voxel_size = cfg.voxel_size
    hi_res_voxel = cfg.hi_res_voxel
    patch_size_pixels = int(voxel_size[0] / hi_res_voxel[0])

    if direction == 'right':
        point_cloud_range = cfg.right_point_cloud_range
    elif direction == 'left':
        point_cloud_range = cfg.left_point_cloud_range

    hi_res_grid_size = _compute_grid_size(hi_res_voxel, point_cloud_range)

    pts_cam_ego, colors, segs_pts, segs_colors  = lift(
        cfg, 
        img, 
        cal_pts, 
        de, 
        T_cam_ego, 
        T_lid_cam, 
        T_rad_cam, 
        plot=False,
        seg_mask=seg,
        bboxes=bboxes
    )

    centers = []

    if seg_method:
        for seg_pts in segs_pts:
            db = DBSCAN(eps=0.3, min_samples=10).fit(seg_pts)
            main_cluster = seg_pts[db.labels_ == 0]  # largest cluster
            centers.append(main_cluster.mean(axis=0))

    elif lid_method:
        for bbox in bboxes:
            human_center = get_human_center(cfg, bbox, img, lidar, T_lid_cam, T_cam_ego, closest_pct=30)
            centers.append(human_center)

    center_spheres = make_sphere_pts(centers, radius=0.15)

    pts_cam_ego = np.concatenate([pts_cam_ego, center_spheres.points], axis=0)
    colors = np.concatenate([colors, center_spheres.colors], axis=0)

    bev_cam_hires    = hard_splat(pts_cam_ego, colors, hi_res_voxel, point_cloud_range, hi_res_grid_size)
    cam_patches      = patchify(bev_cam_hires, patch_size_pixels)

    img_with_radar = overlay_radar_on_image(cfg, img, ego_radar, T_cam_ego)

    plot_bev_comparison(
        cfg,
        img_with_radar,
        ego_radar,
        bev_cam_hires,
        voxel_size,
        point_cloud_range,
        patch_size_pixels,
        'test',
        0,
        manual_save=cfg.finetuning_vis
    )

    scatter_seg_points(segs_pts, segs_colors, center_spheres, os.path.join(cfg.finetuning_vis, 'seg_pts.png'))
    plot_img_and_seg(img, seg, bboxes, os.path.join(cfg.finetuning_vis, 'img_n_seg.png'))

if __name__ == '__main__':
    main()