import hydra
from omegaconf import DictConfig
import logging

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
        j = i + 103
        frame = sync_frames(cfg)[j]
        log.info(f'Processing frame {frame}')
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
        log.info(f'Finished with frame {j}')
        break

if __name__ == '__main__':
    main()