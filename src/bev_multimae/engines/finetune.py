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
from bev_multimae.datasets.data import BEVDataset, collate_radar

log = logging.getLogger(__name__)


def run_finetune(cfg: DictConfig):

    # Load and create dataset
    data_path_right = cfg.processed_data_dir_right
    data_path_left = cfg.processed_data_dir_left

    # DATASET INITIALIZATION
    train_ds_right = BEVDataset(
        data_path_right, split="train", augment=False
    )
    val_ds_right = BEVDataset(
        data_path_right, split="val", augment=False
    )

    train_ds_left = BEVDataset(
        data_path_left, split="train", augment=False
    )
    val_ds_left = BEVDataset(
        data_path_left, split="val", augment=False
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
        collate_fn=collate_radar,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        collate_fn=collate_radar,
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
        gradient_clip_val=1.0,
    )

    os.makedirs(cfg.model_folder, exist_ok=True)

    # Start training
    log.info('Starting finetuning...')
    trainer.fit(model_lightning, train_loader, val_loader)

    log.info('Finetuning completed!')


if __name__ == '__main__':
    main()
