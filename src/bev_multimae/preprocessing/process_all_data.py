import hydra
from omegaconf import DictConfig
import logging
import os
import torch

# Local
from bev_multimae.preprocessing.data_pipe import BEVPipeline
from bev_multimae.preprocessing.sync import sync_frames, load_img
from bev_multimae.visualization.BEV_visualization import plot_bev_comparison, overlay_radar_on_image

log = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    log.info('Initializing pipeline...')
    pipeline = BEVPipeline(cfg)
    log.info('Pipeline initialized')

    for i in range(60):
        # j = i + 103
        j = i + 129
        frame = sync_frames(cfg)[j]
        log.info(f'Processing frame {j}')
        output = pipeline.process(frame)

        img = load_img(frame['cam'])

        log.info('Plotting BEV comparison..')

        img_with_radar = overlay_radar_on_image(cfg, img, output['pts_rad_ego']) 

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

        # Save to processed folder

        save_dir = os.path.join(cfg.processed_data_dir, "train")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{j:06d}.pt")

        torch.save({
            "cam_bev": torch.from_numpy(output["bev_cam_hires"]).float(),
            "radar": output["batch_dict_rad"],  # make sure tensors
            "radar_target": torch.from_numpy(output["bev_radar_target"]).float(),
            "meta": {
                "frame_id": j,
                "voxel_size": cfg.voxel_size,
                "pcr": cfg.point_cloud_range,
            }
        }, save_path)


        log.info(f'Finished with frame {j}')
        break
if __name__ == '__main__':
    main()