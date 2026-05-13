from omegaconf import DictConfig, OmegaConf
import hydra
import os
import numpy as np

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg):

    direction = 'right'
    for event in sorted(os.listdir(cfg.mcap_extract_path)):
        OmegaConf.update(cfg, "camera_info", f"data/raw/mcap_extract/{event}/camera/front_{direction}/camera_info.npz")
        cam_info = np.load(cfg.camera_info)

        if np.all(cam_info['D'] == 0):
            print(event)

if __name__ == '__main__':
    main()