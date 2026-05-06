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

from bev_multimae.visualization.finetuning import plot_img_and_seg, scatter_seg_points
from bev_multimae.preprocessing.camera.lift import lift
from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.preprocessing.get_transforms import get_all_tfs
from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar, load_radar, load_seg, load_bbox
from bev_multimae.preprocessing.radar.radar_process_utils import radar_to_ego
from bev_multimae.visualization.BEV_visualization import overlay_radar_on_image, plot_bev_comparison
from bev_multimae.preprocessing.BEV.splat import hard_splat, patchify


def _merge_radar(paths: list) -> dict:
        frames = [load_radar(p) for p in paths]
        return {k: np.concatenate([f[k] for f in frames]) for k in frames[0].keys()}

def make_sphere_pts(centers, radius=0.1, n=500, color=[0, 0, 1]):
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


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    events = sorted(os.listdir(cfg.mcap_extract_path))
    n_events = len(events)

    direction='right'
    # event = 'evt_0e8QraX8B9UIyxY9' # evening, two people to the right in the frame
    event = 'evt_0e8RO9yx2kWoavOD'   # afternoon, person standing still in front of robot while its driving
    # event = 'evt_0e8RSwkcSts5kEaF'  # Two people further away

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


    frame_idx = 10
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
    point_cloud_range = cfg.point_cloud_range
    hi_res_voxel = cfg.hi_res_voxel
    patch_size_pixels = int(voxel_size[0] / hi_res_voxel[0])
    hi_res_grid_size = _compute_grid_size(hi_res_voxel, cfg.point_cloud_range)

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
    for seg_pts in segs_pts:
        db = DBSCAN(eps=0.3, min_samples=10).fit(seg_pts)
        main_cluster = seg_pts[db.labels_ == 0]  # largest cluster
        centers.append(main_cluster.mean(axis=0))

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
        0
    )


    scatter_seg_points(segs_pts, segs_colors, center_spheres, os.path.join(cfg.finetuning_vis, 'seg_pts.png'))
    plot_img_and_seg(img, seg, os.path.join(cfg.finetuning_vis, 'img_n_seg.png'))

if __name__ == '__main__':
    main()