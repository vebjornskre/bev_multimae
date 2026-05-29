# This file will contain pretraining code
import os
import torch
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset

import hydra
from omegaconf import DictConfig, OmegaConf

from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.adapters.feat_adapt import FeatureAdapter
from bev_multimae.multimae.decoders.recon_decoder import SpatialOutputAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.multimae.model_lightning_with_feat import BevMultiMAELightning
from bev_multimae.datasets.data_with_feat import BEVDataset, collate_radar
from bev_multimae.visualization.predictions_with_feat import viz_preds
from bev_multimae.visualization.viz_augment import viz_augment
from bev_multimae.engines.train_utils import *

log = logging.getLogger(__name__)


def load_matching_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print(ckpt.keys())
    print(ckpt["hyper_parameters"])
    
    old_state = ckpt["state_dict"]

    old_state = {
        k.replace("model.", "", 1): v
        for k, v in old_state.items()
        if k.startswith("model.")
    }

    new_state = model.state_dict()
    loadable = {}
    skipped = []

    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    log.info(f"Loaded matching tensors from {ckpt_path}: {len(loadable)}")
    log.info(f"Skipped tensors: {len(skipped)}")
    log.info(f"Missing keys after partial load: {len(missing)}")
    log.info(f"Unexpected keys after partial load: {len(unexpected)}")

    return model

def run_pretrain(cfg: DictConfig):
    
    # Load and create dataset
    # data_path_right = cfg.processed_data_dir_right
    # data_path_left  = cfg.processed_data_dir_left
    
    data_path_right = "data/processed_2/right"
    data_path_left  = "data/processed_2/left"
    

    # DATASET INITIALIZATION
    try:
        ms = torch.load(os.path.join(cfg.processed_data_dir, 'mean_std.pt'))
        img_mean, img_std, rad_mean, rad_std = (ms['img_mean'], ms['img_std'], ms['rad_mean'], ms['rad_std'])
    except:
        img_mean, img_std = compute_img_stats([cfg.processed_data_dir_right, cfg.processed_data_dir_left])
        rad_mean, rad_std = compute_radar_stats([cfg.processed_data_dir_right, cfg.processed_data_dir_left])
        ms = {
            'img_mean': img_mean,
            'img_std':  img_std,
            'rad_mean': rad_mean,
            'rad_std':  rad_std,
        }
        torch.save(ms, os.path.join(cfg.processed_data_dir, 'mean_std.pt'))

    # Re-init train with same stats
    train_ds_right = BEVDataset(
        data_path_right, split="train", 
        img_mean=img_mean, img_std=img_std,
        rad_mean=rad_mean, rad_std=rad_std,
        augment=cfg.augment, h_flip_rate=cfg.h_flip_rate,
        v_flip_rate=cfg.v_flip_rate, rot_rate=cfg.rot_rate,
        rot_angle=cfg.rot_angle, point_cloud_range=cfg.right_point_cloud_range
        )
    val_ds_right = BEVDataset(
        data_path_right, split="val", 
        img_mean=img_mean, img_std=img_std,
        rad_mean=rad_mean, rad_std=rad_std
        )

    train_ds_left = BEVDataset(
        data_path_left, split="train", 
        img_mean=img_mean, img_std=img_std,
        rad_mean=rad_mean, rad_std=rad_std,
        augment=cfg.augment, h_flip_rate=cfg.h_flip_rate,
        v_flip_rate=cfg.v_flip_rate, rot_rate=cfg.rot_rate,
        rot_angle=cfg.rot_angle, point_cloud_range=cfg.left_point_cloud_range
        )
    val_ds_left = BEVDataset(
        data_path_left, split="val", 
        img_mean=img_mean, img_std=img_std,
        rad_mean=rad_mean, rad_std=rad_std
        )

    train_ds = ConcatDataset([train_ds_right, train_ds_left])
    val_ds = ConcatDataset([val_ds_right, val_ds_left])

    if cfg.viz_augment:
        viz_augment(train_ds, idx=150)

    log.info(f'Number of samples: {len(train_ds)}')

    # Unpack meta data from the training data to be used in the adapters
    meta = train_ds_right.meta


    grid_size = meta['grid_size']
    grid_size_hires = meta['hi_res_grid_size']

    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]
    H_cam, W_cam = ny_hi, nx_hi
    patch_size = (H_cam // ny, W_cam // nx)

    num_point_features = meta['num_point_features']
    num_vfe_features = cfg.num_vfe_features

    dim_tokens = cfg.dim_tokens # embed_dim

    bev_feat_grid_size = meta.get("bev_feat_grid_size", cfg.bev_feat_grid_size)
    bev_feat_channels = cfg.bev_feat_channels

    nx_feat, ny_feat = bev_feat_grid_size[:2]
    feat_patch_size = (ny_feat // ny, nx_feat // nx)

    if ny_feat % ny != 0 or nx_feat % nx != 0:
        raise ValueError(
            f"bev_feat_grid_size {bev_feat_grid_size} must be divisible by token grid {(nx, ny)}"
        )

    context_tasks = ["cam_bev", "radar", "bev_feat"]

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

    cam_decode = SpatialOutputAdapter(
        num_channels=cfg.cam_channels,
        stride_level=1,
        patch_size_full=patch_size,
        image_size=(grid_size_hires[1], grid_size_hires[0]),
        task="cam_bev",
        context_tasks=context_tasks,
        dim_tokens=dim_tokens,
        dim_tokens_enc=dim_tokens,
    )

    rad_decode = SpatialOutputAdapter(
        num_channels=cfg.rad_channels,
        stride_level=1,
        patch_size_full=(1, 1),
        image_size=(grid_size[1], grid_size[0]),
        task="radar",
        context_tasks=context_tasks,
        dim_tokens=dim_tokens,
        dim_tokens_enc=dim_tokens,
    )

    feat_decode = SpatialOutputAdapter(
        num_channels=bev_feat_channels,
        stride_level=1,
        patch_size_full=feat_patch_size,
        image_size=(ny_feat, nx_feat),
        task="bev_feat",
        context_tasks=context_tasks,
        dim_tokens=dim_tokens,
        dim_tokens_enc=dim_tokens,
    )

    output_adapters = {
        "cam_bev": cam_decode,
        "radar": rad_decode,
        "bev_feat": feat_decode,
    }

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_folder,
        filename="best_model_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
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
            "learning_rate":cfg.lr, 
            "batch_size":cfg.batch_size,
            "num_epochs":cfg.max_epochs,
            "optimizer":cfg.optimizer
            }
        wandb_logger.log_hyperparams(hyperparams)
    else:
        wandb_logger = None 

    # training parameters (num_vfe_features is also a training parameter)

    train_loader = DataLoader(
        train_ds, 
        cfg.batch_size, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor = cfg.prefetch_factor,
        collate_fn=collate_radar,
        shuffle=True
        )

    val_loader = DataLoader(
        val_ds, 
        cfg.batch_size, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_radar,
        shuffle=False
        )


    model = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=dim_tokens,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        drop_path_rate = cfg.drop_path_rate,
        drop_rate = cfg.drop_rate,
        attn_drop_rate = cfg.attn_drop_rate
    )

    if cfg.get("load_old_pretrain", False):
        old_ckpt_path = cfg.old_pretrain_checkpoint

        if not old_ckpt_path.endswith(".ckpt"):
            old_ckpt_path = f"{old_ckpt_path}.ckpt"

        if not os.path.isabs(old_ckpt_path):
            old_ckpt_path = os.path.join(cfg.model_folder, old_ckpt_path)

        if not os.path.exists(old_ckpt_path):
            raise FileNotFoundError(f"Old pretrain checkpoint not found: {old_ckpt_path}")

        model = load_matching_weights(model, old_ckpt_path)

    model_lightning = BevMultiMAELightning(
        model=model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_encoded_tokens=cfg.num_encoded_tokens,
        norm_pix=cfg.norm_pix,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        dim_tokens=cfg.dim_tokens,
        warmup_steps=cfg.warmup_steps,
        drop_path_rate = cfg.drop_path_rate,
        drop_rate = cfg.drop_rate,
        attn_drop_rate = cfg.attn_drop_rate,
        data_aug=cfg.augment,
        num_rad_channels=cfg.rad_channels,
        feat_weight=cfg.get("feat_weight", 1.0),
        feat_patch_size=cfg.get("bev_feat_patch_size", 3),
    )

    trainer = Trainer(
        max_epochs = cfg.max_epochs,
        min_epochs = cfg.min_epochs,
        enable_checkpointing=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.model_folder,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=1.0,
    )

    ckpt_path = os.path.join(cfg.model_folder, f'{cfg.best_model}.ckpt')
    log.info(f'Checkpoint exists: {os.path.exists(ckpt_path)} — {ckpt_path}')
    continue_training = cfg.continue_training

    os.makedirs(cfg.model_folder, exist_ok=True)

    if continue_training:
        trainer.fit(model_lightning, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model_lightning, train_loader, val_loader)


    model_lightning.model.eval()
    model_lightning.model.cuda()

    with torch.no_grad():
        batch = next(iter(train_loader))
        torch.manual_seed(42)

        batch["cam_bev"] = batch["cam_bev"].cuda()

        if "bev_feat" in batch:
            batch["bev_feat"] = batch["bev_feat"].cuda()

        for k, v in batch["radar"].items():
            if isinstance(v, torch.Tensor):
                batch["radar"][k] = v.cuda()

        batch["radar_target"] = batch["radar_target"].cuda()

        preds, task_masks = model_lightning.model(
            batch,
            mask_inputs=True,
            num_encoded_tokens=cfg.num_encoded_tokens,
        )

        B = task_masks["cam_bev"].shape[0]
        ny, nx = grid_size[1], grid_size[0]
        ph, pw = patch_size

        cam_mask = task_masks["cam_bev"].reshape(B, ny, nx)
        cam_mask = cam_mask.repeat_interleave(ph, dim=1)\
                        .repeat_interleave(pw, dim=2)\
                        .unsqueeze(1).float()

        img_mean = train_ds_right.img_mean.cuda()
        img_std = train_ds_right.img_std.cuda()

        cam_pred = denorm_img(preds["cam_bev"], img_mean, img_std)
        cam_input = denorm_img(batch["cam_bev"], img_mean, img_std)

        if cam_mask.sum() == 0:
            cam_composite = cam_pred
        else:
            cam_composite = cam_pred * cam_mask + cam_input * (1 - cam_mask)

        plot_preds = {
            "cam_bev": cam_composite,
            "radar": preds["radar"],
        }

        if "bev_feat" in preds:
            plot_preds["bev_feat"] = preds["bev_feat"]

        viz_preds(plot_preds, batch, cfg.plot_folder)

if __name__ == '__main__':
    main()