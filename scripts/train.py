import cProfile
import torch
import hydra
from omegaconf import DictConfig

from bev_multimae.engines.pretrain import run_pretrain


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    if cfg.profile:
        prof = cProfile.Profile()
        prof.enable()
        run_pretrain(cfg)
        prof.disable()
        prof.dump_stats(cfg.profile_path)
    else:
        run_pretrain(cfg)


if __name__ == "__main__":
    main()