import torch
import torch.profiler as profiler
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.finetuning.model_lightning import CenterPointLightning
from bev_multimae.finetuning.centerpoint import (
    TokenToSpatialAdapter,
    CenterPointHead,
    CenterPointDetector,
)
from bev_multimae.datasets.finetuning_data import BEVFineData
from bev_multimae.engines.finetune import collate_finetune
import os


@hydra.main(config_path="../configs", config_name="config_finetune", version_base=None)
def main(cfg: DictConfig):
    # Load normalization stats
    try:
        ms = torch.load(os.path.join(cfg.processed_data_dir, 'mean_std.pt'))
        img_mean, img_std = ms['img_mean'], ms['img_std']
    except:
        img_mean, img_std = None, None

    # Create dataset
    train_ds_right = BEVFineData(
        pretrain_path=cfg.processed_data_dir_right,
        finetune_path=cfg.finetuning_data_dir,
        direction="right",
        split="train",
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=cfg.right_point_cloud_range,
        augment=False,
    )

    loader = DataLoader(
        train_ds_right,
        batch_size=cfg.batch_size,
        num_workers=0,  # Single worker for profiling
        collate_fn=collate_finetune,
        shuffle=False
    )

    # Create model
    meta = train_ds_right.meta
    grid_size = meta['grid_size']
    grid_size_hires = meta['hi_res_grid_size']
    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]
    H_cam, W_cam = ny_hi, nx_hi
    patch_size = (H_cam // ny, W_cam // nx)

    dim_tokens = cfg.dim_tokens
    rad_adapt = RadarAdapter(dim_tokens, grid_size, meta['num_point_features'], cfg.num_vfe_features)
    cam_adapt = CameraAdapter(dim_tokens, cfg.cam_channels, patch_size, grid_size_hires)
    input_adapters = {'radar': rad_adapt, 'cam_bev': cam_adapt}

    encoder = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=None,
        dim_tokens=dim_tokens,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
    )

    token_adapter = TokenToSpatialAdapter(
        dim_tokens=dim_tokens,
        output_channels=cfg.centerpoint_channels,
        include_global=cfg.include_global_token
    )
    detection_head = CenterPointHead(in_channels=cfg.centerpoint_channels)
    detector = CenterPointDetector(token_adapter, detection_head)

    model_lightning = CenterPointLightning(encoder=encoder, detector=detector)
    model_lightning = model_lightning.cuda()

    # Profile one batch
    batch = next(iter(loader))
    batch["cam_bev"] = batch["cam_bev"].cuda()
    for k, v in batch["radar"].items():
        if isinstance(v, torch.Tensor):
            batch["radar"][k] = v.cuda()

    print("Profiling one training step...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        loss = model_lightning.training_step(batch, 0)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("profiling/batch_trace.json")


if __name__ == "__main__":
    main()
