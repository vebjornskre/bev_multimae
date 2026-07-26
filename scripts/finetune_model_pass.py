import logging

import hydra
from omegaconf import DictConfig
import random

# from bev_multimae.engines.finetune_inference import setup_and_infer # ALL MODALITIES
from bev_multimae.engines.finetune_inference_feat_rad import setup_and_infer # ONLY FEAT AND RAD
from bev_multimae.visualization.augment_viz import visualize_augmentations

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config_finetune", version_base=None)
def main(cfg: DictConfig):
    sample_idxs = [random.randint(0, 447) for _ in range(20)]
    # sample_idxs = [cfg.get("sample_idx", 0)]
    # sample_idxs = [0]
    # sample_idxs = [x for x in range(40)]
    sample_idxs = [352, 23, 123, 39, 230]

    for sample_idx in sample_idxs:
        visualize = cfg.get("visualize", True)
        sample_dir = f"reports/finetuning/predictions/{sample_idx}"

        log.info(f"Predicting idx {sample_idx} in the {cfg.get('split', 'val')} set")

        visualize_augmentations(
            cfg,
            sample_idx=sample_idx,
            save_dir=f"{sample_dir}/augmentations",
            angle_deg=45,
        )
        setup_and_infer(
            cfg,
            sample_idx,
            visualize=visualize
        )

        if visualize:
            log.info(f"Visualizations saved to {sample_dir}")

if __name__ == "__main__":
    main()