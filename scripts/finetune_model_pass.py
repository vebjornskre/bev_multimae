import logging

import hydra
from omegaconf import DictConfig

from bev_multimae.engines.finetune_inference import setup_and_infer

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config_finetune", version_base=None)
def main(cfg: DictConfig):
    sample_idx = cfg.get("sample_idx", 0)
    visualize = cfg.get("visualize", True)

    log.info(f"Predicting idx {sample_idx} in the {cfg.get('split', 'val')} set")

    setup_and_infer(cfg, sample_idx, visualize=visualize)

    if visualize:
        log.info('Visualizations saved')

if __name__ == "__main__":
    main()