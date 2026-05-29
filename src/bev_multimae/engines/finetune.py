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
from bev_multimae.multimae.adapters.feat_adapt import FeatureAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.finetuning.model_lightning import CenterPointLightning
from bev_multimae.finetuning.centerpoint.token_adapter import TokenToSpatialAdapter
from bev_multimae.finetuning.centerpoint.model import CenterPointHead, CenterPointDetector
from bev_multimae.datasets.data import collate_radar
from bev_multimae.datasets.finetuning_data import BEVFineData, collate_finetune
import numpy as np

log = logging.getLogger(__name__)


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

    grid_size = meta["grid_size"]
    grid_size_hires = meta["hi_res_grid_size"]

    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]

    H_cam, W_cam = ny_hi, nx_hi
    patch_size = (H_cam // ny, W_cam // nx)

    num_point_features = meta["num_point_features"]
    num_vfe_features = cfg.num_vfe_features
    dim_tokens = cfg.dim_tokens

    bev_feat_grid_size = meta.get("bev_feat_grid_size", cfg.bev_feat_grid_size)
    bev_feat_channels = cfg.bev_feat_channels

    nx_feat, ny_feat = bev_feat_grid_size[:2]
    feat_patch_size = (ny_feat // ny, nx_feat // nx)

    if ny_feat % ny != 0 or nx_feat % nx != 0:
        raise ValueError(
            f"bev_feat_grid_size {bev_feat_grid_size} must be divisible by token grid {(nx, ny)}"
        )

    rad_adapt = RadarAdapter(dim_tokens, grid_size, num_point_features, num_vfe_features)
    cam_adapt = CameraAdapter(dim_tokens, cfg.cam_channels, patch_size, grid_size_hires)
    feat_adapt = FeatureAdapter(
        d_model=dim_tokens,
        channels=bev_feat_channels,
        patch_size=feat_patch_size,
        bev_feat_grid_size=(ny_feat, nx_feat),
    )

    input_adapters = {
        "radar": rad_adapt,
        "cam_bev": cam_adapt,
        "bev_feat": feat_adapt,
    }

    encoder = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=None,
        dim_tokens=dim_tokens,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        drop_path_rate=cfg.drop_path_rate,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
    )

    # Load pretrained encoder checkpoint if provided
    if cfg.pretrained_checkpoint and os.path.exists(cfg.pretrained_checkpoint):
        log.info(f"Loading pretrained encoder from {cfg.pretrained_checkpoint}")
        encoder = load_pretrain(encoder, cfg.pretrained_checkpoint)
        log.info("Pretrained encoder loaded successfully")

    # Freeze encoder if specified
    if cfg.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()  # set eval before wrapping in Lightning
        log.info('Encoder frozen for finetuning')

    # Create detector components
    token_adapter = TokenToSpatialAdapter(
        dim_tokens=dim_tokens,
        output_channels=cfg.centerpoint_channels,
        include_global=cfg.include_global_token
    )
    detection_head = CenterPointHead(
        in_channels=cfg.centerpoint_channels,
        num_backbone_layers=cfg.get("num_backbone_layers", 2),
        dropout=cfg.get("centerpoint_dropout", 0.0),
    )
    detector = CenterPointDetector(token_adapter, detection_head)

    detector = torch.compile(detector)

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.fine_model_folder,
        filename="best_3Dmodel_{epoch:02d}_{val_total_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=cfg.save_top_k,
        save_last=False,
    )

    # WandB logging
    if cfg.wandb_project_finetune:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project_finetune,
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
        encoder_lr=cfg.get("encoder_lr", cfg.lr),
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        heatmap_weight=cfg.heatmap_weight,
        offset_weight=cfg.offset_weight,
        height_weight=cfg.height_weight,
        dim_weight=cfg.dim_weight,
        rot_weight=cfg.rot_weight,
        modality_dropout=cfg.get("modality_dropout", False),
        drop_radar_prob=cfg.get("drop_radar_prob", 0.0),
        drop_cam_prob=cfg.get("drop_cam_prob", 0.0),
        drop_feat_prob=cfg.get("radar_feat_prob", 0.0),
        freeze_encoder=cfg.freeze_encoder
    )

    if cfg.get("continue_training", False):
        ckpt_path = os.path.join(cfg.fine_model_folder, cfg.get("continue_checkpoint"))
        log.info(f"Continuing from finetuned checkpoint weights: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_lightning.load_state_dict(ckpt["state_dict"], strict=False)

        log.info("Finetuned model weights loaded")

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
        enable_progress_bar=False,
        precision=cfg.get("precision", "16-mixed")
    )

    os.makedirs(cfg.model_folder, exist_ok=True)

    # Start training
    log.info('Starting finetuning...')
    trainer.fit(model_lightning, train_loader, val_loader)

    log.info('Finetuning completed!')


if __name__ == '__main__':

    @hydra.main(config_path="../../configs", config_name="config_finetune", version_base=None)
    def main(cfg: DictConfig):
        run_finetune(cfg)

    main()
