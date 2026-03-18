import matplotlib.pyplot as plt
import numpy as np
import logging
import open3d as o3d
import hydra
import os
from pathlib import Path
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def plot_lifted_points(cfg, pts, colors, img, meshlab=False):

    # Save folder from cfg
    plot_folder = os.path.join(Path(to_absolute_path(cfg.plot_folder)), 'point_clouds')
    fname = os.path.join(f'lifted_points_{cfg.depth_model}.png')
    save_path = os.path.join(plot_folder, fname)

    valid = ~np.isnan(pts).any(axis=1)

    # Downsampling
    step = 1
    pts, colors = pts[valid][::step], colors[valid][::step]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(img)
    axes[0].set_title('Input image')
    axes[0].axis('off')

    axes[1].scatter(pts[:, 1], pts[:, 0], s=0.3, c=colors)
    axes[1].set_xlabel('Left / Right (m)')
    axes[1].set_ylabel('Depth (m)')
    axes[1].set_title('BEV (top-down)')
    axes[1].set_aspect('equal')
    axes[1].invert_xaxis()

    axes[2].scatter(pts[:, 1], pts[:, 2], s=0.3, c=colors)
    axes[2].set_xlabel('Left / Right (m)')
    axes[2].set_ylabel('Height (m)')
    axes[2].set_title('Front view')
    axes[2].set_aspect('equal')
    axes[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    log.info(f"Saved {fname}")

    if meshlab:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(f'{save_path[:-3]}ply', pcd)

if __name__ == '__main__':
    print('This file can not be run as a stand alone script at this time')