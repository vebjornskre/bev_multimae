import cv2
import numpy as np
import os

import hydra
from omegaconf import DictConfig

from bev_multimae.preprocessing.sync import sync_frames, load_img, load_lidar, load_radar



@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    cam_info = np.load(cfg.camera_info)
    intrinsic_matrix, D = cam_info["K"], cam_info["D"]

    print(intrinsic_matrix)
    print(D)

    # From Rolando Teams: lidar_front_top/laser → camera_front_right_optical_frame
    T_lid = np.array([
        [0.396, -0.266,  0.879, -0.056],
        [0.917,  0.072, -0.391,  0.504],
        [0.041, 0.961,  0.273,  0.282],
        [0.000,  0.000,  0.000,  1.000]
    ])

    # From Rolando Teams: camera_front_right_optical_frame  → lidar_front_top/laser
    T_lid = np.linalg.inv(T_lid)

    # From Rolando Teams: radar_front_right/laser → camera_front_right_optical_frame
    T_rad = np.array([
        [-0.030, -0.162, 0.986, -0.078],
        [0.999, 0.003, 0.031, 0.002],
        [-0.009, 0.987, 0.162, 0.108],
        [0.000,  0.000,  0.000,  1.000]
    ])

    # From Rolando Teams: camera_front_right_optical_frame  → radar_front_right
    T_rad = np.linalg.inv(T_rad)

    R = T_rad[:3, :3]
    t = T_rad[:3, 3]

    print(f'R_mat: {R}')
    print(f'T_vec: {t}')

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)


    save_folder = os.path.join(cfg.plot_folder, 'projection_test')


    for j in range(170):

        frame = sync_frames(cfg)[j]

        img = load_img(frame['cam'])
        W, H = img.size
        img = np.array(img)
        # img = cv2.undistort(img, intrinsic_matrix, D)
        
        lidar = load_lidar(frame['lid'])
        frames = [load_radar(p) for p in frame['rad']]
        rad = {k: np.concatenate([f[k] for f in frames]) for k in frames[0].keys()}

        points_3d = lidar
        points_3d = np.stack([rad["x"], rad["y"], rad["z"]], axis=1).astype(np.float32)

        # Project 3D points onto 2D plane
        points_2d, _ = cv2.projectPoints(points_3d,
                                        rvec, tvec.reshape(-1, 1),
                                        intrinsic_matrix,
                                        D)
        

        # Save figure
        save_path   = os.path.join(save_folder, f'projection_test_{j}.png')

        print(j)
        
        i = 0
        # Plot 2D points
        for point in points_2d.astype(int):
            
            x = point[0][0]
            y = point[0][1]

            if x < 0 or x > W:
                continue
            if y < 0 or y > H:
                continue
            img = cv2.circle(img, (x,y), 2, 255, -1)


            i += 1


        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main()