import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch

from bev_multimae.pipelines.data_pipe import BEVPipeline
from bev_multimae.preprocessing.sync import sync_frames, load_img
from bev_multimae.visualization.BEV_visualization import plot_bev_comparison, overlay_radar_on_image

log = logging.getLogger(__name__)

# EVENT = "evt_0e8RBjD19OTccsWm"
# EVENT = "evt_0e3qa9akdU4BIHaF" 
# EVENT = "evt_0e8Qmgb6bdukqOwj"
# EVENT = "evt_0e8Qot9N5B051LcL"
EVENT = "evt_0dyCcjgqVowqWW3y"
FRAME_IDX = 0              

# evt_0e3rMOlUlNNs8CD1

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    direction = cfg.direction
    if direction == 'left':
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/left")
    if direction == 'right':
        OmegaConf.update(cfg, "processed_data_dir", "data/processed/right")

    OmegaConf.update(cfg, "camera_info", f"data/raw/mcap_extract/{EVENT}/camera/front_{direction}/camera_info.npz")
    OmegaConf.update(cfg, "radar_raw_path", f"data/raw/mcap_extract/{EVENT}/radar/front_{direction}")
    OmegaConf.update(cfg, "imgs_raw_path", f"data/raw/mcap_extract/{EVENT}/camera/front_{direction}")
    OmegaConf.update(cfg, "lidar_raw_path", f"data/raw/mcap_extract/{EVENT}/lidar/front_top")
    OmegaConf.update(cfg, "mcap_path", os.path.join(cfg.bags_path, f"{EVENT}.mcap"))

    frames = sync_frames(cfg)
    frame = frames[FRAME_IDX]
    log.info(f'Using event: {EVENT}, frame: {FRAME_IDX}/{len(frames)}')

    # models = ['metric3d', 'depth_any', 'moge', 'depth_any_rel', 'depth_pro', 'unidepth'] 
    # models = ['zoe']
    models = ['moge']

    for m in models:
        OmegaConf.update(cfg, "depth_model", m)

        log.info('Initializing pipeline...')
        pipeline = BEVPipeline(cfg)
        log.info('Pipeline initialized')

        output = pipeline.process(frame)

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
            EVENT,
            FRAME_IDX,
            manual_save = os.path.join(f'reports/figures/depth_imgs/bev_{m}')
        )

        print(output['bev_cam_hires'].shape)

    return output, pipeline, frame

if __name__ == '__main__':
    main()