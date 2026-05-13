import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import hydra
from omegaconf import DictConfig, OmegaConf

from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar, load_radar
from bev_multimae.preprocessing.get_transforms import get_all_tfs
from bev_multimae.preprocessing.get_transforms import apply_transform


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    events = sorted(os.listdir(cfg.mcap_extract_path))
    n_events = len(events)

    direction='right'
    seg_method = False
    lid_method = not seg_method

    # event = 'evt_0e8QraX8B9UIyxY9' # evening, two people to the right in the frame
    # event = 'evt_0e8RO9yx2kWoavOD'   # afternoon, person standing still in front of robot while its driving
    # event = 'evt_0e8RSwkcSts5kEaF'  # Two people further away
    # event = 'evt_0e3qa9akdU4BIHaF' 
    # event = 'evt_0e3rMglACswVRZ1U'    # At night one person
    event = 'evt_0e3qZe9bbrfW2dr6'

    # Bags that lack distortion coeficients
    # event = 'evt_0dyEeORn8jJHCOq2' ## -3.36873119e-01
    # event = 'evt_0dz4DPX49GHES3QR' ## -3.16873119e-01
    # event = 'evt_0e8QmDh0sX27pfVW' ## -3.36873119e-01
    # event = 'evt_0e8QmOXkpfZKY1ih' ## -3.36873119e-01
    # event = 'evt_0e8QmXqaenvFbugE' ## -3.36873119e-01
    # event = 'evt_0e8Qmgb6bdukqOwj' ## -2.36873119e-01
    # event = 'evt_0e8QmsFL0xUEIEWj' ## -2.36873119e-01
    # event = 'evt_0e8Qn5kqXWs3r28T' ## -2.36873119e-01
    # event = 'evt_0e8QnBKXiE4mtDNl' ## -2.76873119e-01
    # event = 'evt_0e8QndIYxPSnB0y9' ## -2.76873119e-01

    frame_idx = 0  

    OmegaConf.update(cfg, "camera_info", f"data/raw/mcap_extract/{event}/camera/front_{direction}/camera_info.npz")
    OmegaConf.update(cfg, "radar_raw_path", f"data/raw/mcap_extract/{event}/radar/front_{direction}")
    OmegaConf.update(cfg, "imgs_raw_path", f"data/raw/mcap_extract/{event}/camera/front_{direction}")
    OmegaConf.update(cfg, "seg_raw_path", f"data/raw/mcap_extract/{event}/seg/front_{direction}")
    OmegaConf.update(cfg, "bbox_raw_path", f"data/raw/mcap_extract/{event}/bbox/front_{direction}")
    OmegaConf.update(cfg, "lidar_raw_path", f"data/raw/mcap_extract/{event}/lidar/front_top")
    OmegaConf.update(cfg, "mcap_path", f"data/raw/bags/{event}.mcap")


    with open(os.path.join(f"data/raw/mcap_extract/{event}", "timestamp_info.txt")) as txt:
        lines = txt.readlines()

        if lines[8][-36:-28] == 'log_time':
            print('YOOOOOOOOOO')


    lidar_path = os.path.join(cfg.mcap_extract_path, event, "lidar", "front_top")
    if not os.path.exists(lidar_path) or not os.listdir(lidar_path):
        print(f'Skipping {event} — empty or missing lidar folder')

    radar_path = os.path.join(cfg.mcap_extract_path, event, "radar", f"front_{direction}")
    if not os.path.exists(radar_path) or not os.listdir(radar_path):
        print(f'Skipping {event} — empty or missing radar folder')

    frames = sync_frames(cfg, seg=True)

    T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, T_lid_ego = get_all_tfs(cfg, right=True)

    R = T_lid_cam[:3, :3]
    t = T_lid_cam[:3, 3]

    print(f'R_mat: {R}')
    print(f'T_vec: {t}')

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)


    save_folder = os.path.join(cfg.plot_folder, 'projection_test')

    cam_info = dict(np.load(cfg.camera_info))

    if np.all(cam_info['D'] == 0):
        print('HELLO THERES NO DISTORTION IN THIS BAG')
        cam_info['D'] = np.array([-3.36873119e-01, 1.29256173e-01, 1.02774231e-03, 1.23003590e-04, -2.42683235e-02])
        # cam_info['D'] = np.array([-2.76873119e-01, 1.29256173e-01, 1.02774231e-03, 1.23003590e-04, -2.42683235e-02])

    intrinsic_matrix, D = cam_info['K'], cam_info['D']

    for j, frame in enumerate(frames):

        img = load_img(frame['cam'])
        W, H = img.size
        img = np.array(img)
        # img = cv2.undistort(img, intrinsic_matrix, D)
        
        lidar = load_lidar(frame['lid'])
        frames = [load_radar(p) for p in frame['rad']]
        rad = {k: np.concatenate([f[k] for f in frames]) for k in frames[0].keys()}

        points_3d = apply_transform(T_lid_cam, lidar)

        # points_3d = np.stack([rad["x"], rad["y"], rad["z"]], axis=1).astype(np.float32)

        z = points_3d[:, 2]

        lateral = points_3d[:, 0]
        depth = points_3d[:, 2]
        height = points_3d[:, 1]

        valid = (
            (np.abs(lateral / depth) < 1.4)  
        )
        points_3d = points_3d[valid]

        # Project 3D points onto 2D plane
        points_2d, _ = cv2.projectPoints(points_3d,
                                        np.zeros(3), np.zeros(3),
                                        intrinsic_matrix,
                                        D)
        
        # Normalize distances for coloring
        distances = points_3d[:, 2]
        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)
        cmap = cm.get_cmap('jet')

        save_path = os.path.join(save_folder, f'projection_test_{j}.png')
        print(j)
        
        # Plot 2D points with colors based on distance
        for point, dn in zip(points_2d.astype(int), dist_norm):
            x, y = point[0][0], point[0][1]
            if 0 <= x < W and 0 <= y < H:
                color_bgr = tuple(int(c * 255) for c in reversed(cmap(dn)[:3]))
                img = cv2.circle(img, (x, y), 2, color_bgr, -1)


        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main()