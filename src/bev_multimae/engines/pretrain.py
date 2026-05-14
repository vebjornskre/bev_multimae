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
from bev_multimae.multimae.decoders.recon_decoder import SpatialOutputAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.multimae.model_lightning import BevMultiMAELightning
from bev_multimae.datasets.data import BEVDataset, collate_radar
from bev_multimae.visualization.predictions import viz_preds
from bev_multimae.visualization.viz_augment import viz_augment
from bev_multimae.engines.train_utils import *

log = logging.getLogger(__name__)


def run_pretrain(cfg: DictConfig):
    
    # Load and create dataset
    data_path_right = cfg.processed_data_dir_right
    data_path_left  = cfg.processed_data_dir_left

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

    rad_adapt = RadarAdapter(dim_tokens, grid_size, num_point_features, num_vfe_features)
    cam_adapt = CameraAdapter(dim_tokens, cfg.cam_channels, patch_size, grid_size_hires)
    input_adapters = {
        'radar' : rad_adapt,
        'cam_bev'   : cam_adapt
    }

    # print("grid_size_hires:", grid_size_hires)
    # print("patch_size:", patch_size)
    # print("H_cam, W_cam:", H_cam, W_cam)
    # print("decoder h_posemb, w_posemb:", H_cam // patch_size[0], W_cam // patch_size[1])

    cam_decode = SpatialOutputAdapter(
        num_channels=cfg.cam_channels,
        stride_level=1,
        patch_size_full=patch_size,
        image_size=(grid_size_hires[1], grid_size_hires[0]),
        task='cam_bev',
        context_tasks=['cam_bev', 'radar'],
        dim_tokens=dim_tokens,      
        dim_tokens_enc=dim_tokens
    )

    # print("pos_emb shape:", cam_decode.pos_emb.shape) 

    # print("proj_context:", cam_decode.proj_context.weight.shape)
    # print("decoder dim_tokens:", cam_decode.dim_tokens)
    # print("out_proj:", cam_decode.out_proj.weight.shape)

    rad_decode = SpatialOutputAdapter(
        num_channels=cfg.rad_channels,
        stride_level=1,
        patch_size_full=(1, 1),  # each token is already one grid cell
        image_size=(grid_size[1], grid_size[0]),
        task='radar',
        context_tasks=['cam_bev', 'radar'],
        dim_tokens_enc=dim_tokens
    )

    output_adapters = {
        'cam_bev': cam_decode,
        'radar': rad_decode, 
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
        num_rad_channels=cfg.rad_channels
    )

    trainer = Trainer(
        max_epochs = cfg.max_epochs,
        min_epochs = cfg.min_epochs,
        enable_checkpointing=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.model_folder,
        log_every_n_steps=len(train_loader),
        # log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=1.0,
    )

    ckpt_path = os.path.join(cfg.model_folder, f'{cfg.best_model}.ckpt')
    log.info(f'Checkpoint exists: {os.path.exists(ckpt_path)} — {ckpt_path}')
    continue_training = cfg.continue_training

    os.makedirs(cfg.model_folder, exist_ok=True)


    if continue_training and os.path.exists(ckpt_path):
        log.info(f'Continue training of {cfg.best_model}')

        ckpt = torch.load(ckpt_path, map_location="cpu")
        hp = ckpt["hyper_parameters"]

        try:
            num_rad_channels = hp['num_rad_channels']
        except:
            num_rad_channels = 9
    
        input_adapters = {
            "radar": RadarAdapter(hp['dim_tokens'], grid_size, meta['num_point_features'], cfg.num_vfe_features),
            "cam_bev": CameraAdapter(hp['dim_tokens'], cfg.cam_channels, patch_size, grid_size_hires),
        }
        output_adapters = {
            "cam_bev": SpatialOutputAdapter(
                num_channels=cfg.cam_channels, stride_level=1,
                patch_size_full=patch_size, image_size=(grid_size_hires[1], grid_size_hires[0]),
                task="cam_bev", context_tasks=["cam_bev", "radar"],
                dim_tokens=hp['dim_tokens'], dim_tokens_enc=hp['dim_tokens'],
            ),
            "radar": SpatialOutputAdapter(
                num_channels=num_rad_channels, stride_level=1,
                patch_size_full=(1, 1), image_size=(grid_size[1], grid_size[0]),
                task="radar", context_tasks=["cam_bev", "radar"],
                dim_tokens_enc=hp['dim_tokens'],
            ),
        }
        model = Bev_MultiMAE(
            input_adapters=input_adapters, output_adapters=output_adapters,
            dim_tokens=hp["dim_tokens"], depth=hp["depth"], num_heads=hp["num_heads"],
        )

        if cfg.new_lr:
            # weights only, fresh optimizer/scheduler with cfg.lr
            model_lightning = BevMultiMAELightning.load_from_checkpoint(
                ckpt_path, model=model, lr=cfg.lr, strict=False, norm_pix=cfg.norm_pix
            )
            trainer.fit(model_lightning, train_loader, val_loader)
        else:
            # full resume, optimizer and scheduler state restored
            model_lightning = BevMultiMAELightning.load_from_checkpoint(ckpt_path, model=model)

            trainer.fit(model_lightning, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model_lightning, train_loader, val_loader)


    model_lightning.model.eval()
    model_lightning.model.cuda()

    with torch.no_grad():
        batch = next(iter(train_loader))
        torch.manual_seed(42)

        batch["cam_bev"] = batch["cam_bev"].cuda()
        for k, v in batch["radar"].items():
            if isinstance(v, torch.Tensor):
                batch["radar"][k] = v.cuda()
        batch["radar_target"] = batch["radar_target"].cuda()

        preds, task_masks = model_lightning.model(
            batch,
            mask_inputs=True, 
            num_encoded_tokens=cfg.num_encoded_tokens
        )

        # build mask
        B = task_masks["cam_bev"].shape[0]
        ny, nx = grid_size[1], grid_size[0]
        ph, pw = patch_size

        cam_mask = task_masks["cam_bev"].reshape(B, ny, nx)
        cam_mask = cam_mask.repeat_interleave(ph, dim=1)\
                        .repeat_interleave(pw, dim=2)\
                        .unsqueeze(1).float()


        # composite logic
        if cam_mask.sum() == 0:
            composite = preds["cam_bev"]  
        else:
            composite = preds["cam_bev"] * cam_mask + batch["cam_bev"] * (1 - cam_mask)

        img_mean = train_ds_right.img_mean.cuda()
        img_std  = train_ds_right.img_std.cuda()

        cam_pred  = denorm_img(preds["cam_bev"], img_mean, img_std)
        cam_input = denorm_img(batch["cam_bev"], img_mean, img_std)
    
        if cam_mask.sum() == 0:
            composite = cam_pred
        else:
            composite = cam_pred * cam_mask + cam_input * (1 - cam_mask)

        viz_preds(
            {"cam_bev": composite, "radar": preds["radar"]},
            batch,
            cfg.plot_folder
        )

if __name__ == '__main__':
    main()