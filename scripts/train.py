import cProfile
import torch
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    if cfg.use_bev_feat:
        from bev_multimae.engines.finetune import run_finetune
        from bev_multimae.engines.pretrain_with_features import run_pretrain

        print('Using fetures as a modality')
    else:
        from bev_multimae.engines.finetune_2_modalities import run_finetune
        from bev_multimae.engines.pretrain import run_pretrain

        print('Only camBEV and radar')

    torch.set_float32_matmul_precision(cfg.matmul_precision)

    train_mode = cfg.get("train_mode", "finetune")

    if train_mode == 'pretrain':
        if cfg.profile:
            prof = cProfile.Profile()
            prof.enable()
            run_pretrain(cfg)
            prof.disable()
            prof.dump_stats(cfg.profile_path)
        else:
            run_pretrain(cfg)

    elif train_mode == 'finetune':
        prof = cProfile.Profile()
        prof.enable()
        run_finetune(cfg)
        prof.disable()
        prof.dump_stats(cfg.profile_path)

if __name__ == "__main__":
    main()