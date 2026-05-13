import os
import logging

import hydra
from omegaconf import DictConfig

from bev_multimae.engines.inference import setup_and_infer


log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    sample_idx = 1200
    log.info(f'Predicting idx {sample_idx} in the validation set')

    diagnostic = True
    diag_mode  = "rad_patch_probe" # same radar but add camera patches form different modalities

    visualize  = True

    prediction, target = setup_and_infer(cfg, sample_idx, visualize=visualize, diagnostic=diagnostic, diag_mode=diag_mode)


if __name__ == "__main__":
    main()