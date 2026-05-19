# Finetuning code for CenterPoint object detection
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig, OmegaConf

from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.finetuning.model_lightning import CenterPointLightning
from bev_multimae.finetuning.centerpoint import (
    TokenToSpatialAdapter,
    CenterPointHead,
    CenterPointDetector,
)
from bev_multimae.datasets.data import collate_radar
from bev_multimae.datasets.finetuning_data import BEVFineData
import numpy as np

log = logging.getLogger(__name__)


def collate_finetune(batch):
    """Collate function for finetuning - keeps boxes, targets built on GPU."""
    # Collate radar using pillar format
    radar_list = [item['radar'] for item in batch]
    radar_batch = {}

    for k in radar_list[0].keys():
        if k == "batch_size":
            continue
        radar_batch[k] = torch.cat([r[k] for r in radar_list], dim=0)

    # Remap batch indices for points and pillar_coords
    points_list, coords_list = [], []
    for i, r in enumerate(radar_list):
        p = r["points"].clone()
        p[:, 0] = i
        points_list.append(p)
        c = r["pillar_coords"].clone()
        c[:, 0] = i
        coords_list.append(c)

    pillar_offset = 0
    inv_list = []
    for i, r in enumerate(radar_list):
        inv = r["pillar_inv"].clone() + pillar_offset
        inv_list.append(inv)
        pillar_offset += r["pillar_coords"].shape[0]

    radar_batch["pillar_inv"] = torch.cat(inv_list)
    radar_batch["points"] = torch.cat(points_list)
    radar_batch["pillar_coords"] = torch.cat(coords_list)
    radar_batch["batch_size"] = len(batch)

    # Stack camera images
    cam_bev = torch.stack([item['cam_bev'] for item in batch])

    # Keep boxes as-is (will build targets on GPU in training_step)
    boxes_list = [item['boxes'] for item in batch]

    return {
        'cam_bev': cam_bev,
        'radar': radar_batch,
        'boxes': boxes_list,
    }


def run_finetune(cfg: DictConfig):

    # Load and create dataset
    pretrain_path_right = cfg.processed_data_dir_right
    pretrain_path_left = cfg.processed_data_dir_left
    finetune_path = cfg.finetuning_data_dir

    # Load normalization stats from pretraining
    try:
        ms = torch.load(os.path.join(cfg.processed_data_dir, 'mean_std.pt'))
        img_mean, img_std = ms['img_mean'], ms['img_std']
    except:
        img_mean, img_std = None, None
        log.warning('Could not load normalization stats, using None')

    # DATASET INITIALIZATION
    train_ds_right = BEVFineData(
        pretrain_path=pretrain_path_right,
        finetune_path=finetune_path,
        direction="right",
        split="train",
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=cfg.right_point_cloud_range,
        augment=cfg.augment,
    )
    val_ds_right = BEVFineData(
        pretrain_path=pretrain_path_right,
        finetune_path=finetune_path,
        direction="right",
        split="val",
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=cfg.right_point_cloud_range,
        augment=False,
    )

    train_ds_left = BEVFineData(
        pretrain_path=pretrain_path_left,
        finetune_path=finetune_path,
        direction="left",
        split="train",
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=cfg.left_point_cloud_range,
        augment=cfg.augment,
    )
    val_ds_left = BEVFineData(
        pretrain_path=pretrain_path_left,
        finetune_path=finetune_path,
        direction="left",
        split="val",
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=cfg.left_point_cloud_range,
        augment=False,
    )

    train_ds = ConcatDataset([train_ds_right, train_ds_left])
    val_ds = ConcatDataset([val_ds_right, val_ds_left])

    log.info(f'Number of samples: {len(train_ds)}')

    # Unpack meta data from the training data
    meta = train_ds_right.meta

    grid_size = meta['grid_size']
    grid_size_hires = meta['hi_res_grid_size']

    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]
    H_cam, W_cam = ny_hi, nx_hi
    patch_size = (H_cam // ny, W_cam // nx)

    num_point_features = meta['num_point_features']
    num_vfe_features = cfg.num_vfe_features
    dim_tokens = cfg.dim_tokens

    # Create input adapters for encoder
    rad_adapt = RadarAdapter(dim_tokens, grid_size, num_point_features, num_vfe_features)
    cam_adapt = CameraAdapter(dim_tokens, cfg.cam_channels, patch_size, grid_size_hires)
    input_adapters = {
        'radar': rad_adapt,
        'cam_bev': cam_adapt
    }

    # Create encoder (pretrained)
    encoder = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=None,
        dim_tokens=dim_tokens,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        drop_path_rate=cfg.drop_path_rate,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate
    )

    # Load pretrained encoder checkpoint if provided
    if cfg.pretrained_checkpoint and os.path.exists(cfg.pretrained_checkpoint):
        log.info(f'Loading pretrained encoder from {cfg.pretrained_checkpoint}')
        ckpt = torch.load(cfg.pretrained_checkpoint, map_location='cpu')

        # Extract model weights from Lightning checkpoint
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            encoder.load_state_dict(state_dict, strict=False)
        else:
            encoder.load_state_dict(ckpt, strict=False)

        log.info('Pretrained encoder loaded successfully')

    # Freeze encoder if specified
    if cfg.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        log.info('Encoder frozen for finetuning')

    # Create detector components
    token_adapter = TokenToSpatialAdapter(
        dim_tokens=dim_tokens,
        output_channels=cfg.centerpoint_channels,
        include_global=cfg.include_global_token
    )
    detection_head = CenterPointHead(in_channels=cfg.centerpoint_channels)
    detector = CenterPointDetector(token_adapter, detection_head)

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_folder,
        filename="best_model_{epoch:02d}_{val/total_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=cfg.save_top_k,
        save_last=False,
    )

    # WandB logging
    if cfg.wandb_project:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            log_model=False,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        hyperparams = {
            "learning_rate": cfg.lr,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.max_epochs,
            "optimizer": cfg.optimizer,
            "freeze_encoder": cfg.freeze_encoder,
        }
        wandb_logger.log_hyperparams(hyperparams)
    else:
        wandb_logger = None

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        collate_fn=collate_finetune,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        collate_fn=collate_finetune,
        shuffle=False
    )

    # Create Lightning module
    model_lightning = CenterPointLightning(
        encoder=encoder,
        detector=detector,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        num_encoded_tokens=cfg.num_encoded_tokens,
        heatmap_weight=cfg.heatmap_weight,
        offset_weight=cfg.offset_weight,
        height_weight=cfg.height_weight,
        dim_weight=cfg.dim_weight,
        rot_weight=cfg.rot_weight,
    )

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        enable_checkpointing=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.model_folder,
        log_every_n_steps=len(train_loader),
        gradient_clip_val=1.0
    )

    os.makedirs(cfg.model_folder, exist_ok=True)

    # Start training
    log.info('Starting finetuning...')
    trainer.fit(model_lightning, train_loader, val_loader)

    log.info('Finetuning completed!')


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../configs", config_name="config_finetune", version_base=None)
    def main(cfg: DictConfig):
        run_finetune(cfg)

    main()
