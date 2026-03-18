import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from bev_multimae.preprocessing.radar.radar_process_utils import load_and_process, get_radar_transform, build_thresholds


def visualize_radar_scan(radar, plot_folder, ply=False):

    # x=forward, y=left, z=up
    x = radar["x"]
    y = -radar["y"]
    z = radar["z"]
    rcs = radar["radar_cross_section"]

    # BEV plot
    fig = plt.figure(figsize=(7,6))
    sc = plt.scatter(y, x, c=rcs, s=0.5, cmap="viridis")
    plt.xlabel("Left (m)")
    plt.ylabel("Forward (m)")
    plt.title("Radar BEV")
    plt.gca().set_aspect("equal")
    plt.colorbar(sc, label="RCS (dBsm)")
    plt.tight_layout()
    bev_path = plot_folder / "radar_bev.png"
    plt.savefig(bev_path, dpi=200)
    plt.close()

    # Front view plot
    fig = plt.figure(figsize=(10,4))
    sc = plt.scatter(y, z, c=rcs, s=0.5, cmap="viridis")
    plt.xlabel("Left (m)")
    plt.ylabel("Up (m)")
    plt.title("Radar Front View")
    plt.gca().set_aspect("equal")
    plt.colorbar(sc, label="RCS (dBsm)")
    plt.tight_layout()
    front_path = plot_folder / "radar_front.png"
    plt.savefig(front_path, dpi=200)
    plt.close()

    if ply:
    # Export .ply
        import open3d as o3d
        pts = np.stack([x, y, z], axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        norm_rcs = (rcs - np.min(rcs)) / (np.max(rcs) - np.min(rcs) + 1e-6)
        colors = plt.cm.viridis(norm_rcs)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        ply_path = plot_folder / "radar_points.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved {ply_path}")


    print(f"Saved {bev_path}")
    print(f"Saved {front_path}")


@hydra.main(config_path="../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig):
    radar_dir = to_absolute_path(cfg.radar_raw_path)
    frame = cfg.radar_frame

    radar_file = os.listdir(radar_dir)[frame]
    path = os.path.join(radar_dir, radar_file)
    
    plot_folder = Path(to_absolute_path(cfg.plot_folder)) / "point_clouds"

    thresholds = build_thresholds(cfg)
    T = get_radar_transform(cfg.mcap_path)
    
    radar = load_and_process(path, thresholds, T)

    visualize_radar_scan(radar, plot_folder, ply=True)


if __name__ == "__main__":
    main()