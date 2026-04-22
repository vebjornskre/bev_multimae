import hydra
from omegaconf import DictConfig
import logging
import os
import torch
import cProfile

# Local
from bev_multimae.pipelines.data_pipe import BEVPipeline
from bev_multimae.preprocessing.sync import sync_frames, load_img
from bev_multimae.visualization.BEV_visualization import plot_bev_comparison, overlay_radar_on_image

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    prof = cProfile.Profile()

    log.info('Initializing pipeline...')

    pipeline = BEVPipeline(cfg)
    
    log.info('Pipeline initialized')

    save_dir = os.path.join(cfg.processed_data_dir, "train")
    os.makedirs(save_dir, exist_ok=True)

    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0

    num_feats = None

    frames = sync_frames(cfg)

    prof.enable()

    for i in range(78):
        # j = i + 102
        j = i
        frame = frames[j]

        log.info(f'Processing frame {j}')
        output = pipeline.process(frame)

        ego = output["pts_rad_ego"]

        if num_feats is None:
            num_feats = ego.shape[1]

        x = torch.from_numpy(ego[:, 3:6]).float()

        sum_ += x.sum(0)
        sum_sq += (x * x).sum(0)
        count += x.shape[0]

        if cfg.plotting:
            img = load_img(frame['cam'])
            img_with_radar = overlay_radar_on_image(cfg, img, output['pts_rad_ego'], pipeline.T_cam_ego)

            plot_bev_comparison(
                cfg,
                img_with_radar,
                output['pts_rad_ego'],
                output['bev_cam_hires'],
                pipeline.voxel_size,
                pipeline.point_cloud_range,
                pipeline.patch_size_pixels,
                j
            )

        save_path = os.path.join(save_dir, f"{j:06d}.pt")

        torch.save({
            "cam_bev": output["bev_cam_hires"].float(),
            "radar": output["batch_dict_rad"],
            "radar_target": output["bev_radar_target"].float(),
        }, save_path)

        log.info(f'Finished frame {j}')

        if i == 50:
            break

    prof.disable()
    prof.dump_stats("profile.prof")

    mean = sum_ / count
    std = torch.sqrt(sum_sq / count - mean * mean)

    torch.save({
        "num_rad_channels": pipeline.num_rad_channels,
        "num_cam_channels": pipeline.num_cam_channels,
        "voxel_size": pipeline.voxel_size,
        "pcr": pipeline.point_cloud_range,
        "grid_size": pipeline.grid_size,
        "hi_res_grid_size": pipeline.hi_res_grid_size,
        "patch_size": pipeline.patch_size_pixels,
        "num_point_features": num_feats,
        "radar_mean": mean,
        "radar_std": std,
    }, os.path.join(cfg.processed_data_dir, "meta.pt"))

if __name__ == '__main__':
    main()