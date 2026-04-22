from pytorch_lightning import Trainer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from functools import partial

# Hydra
import hydra
from omegaconf import DictConfig

# local
from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.decoders.recon_decoder import SpatialOutputAdapter

from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.multimae.model_lightning import BevMultiMAELightning
from bev_multimae.datasets.data import BEVDataset, collate_radar

from bev_multimae.visualization.predictions import viz_preds

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load and create dataset
    data_path = cfg.processed_data_dir
    train_dataset = BEVDataset(data_path, cfg)

    # Unpack meta data from the training data to be used in the adapters
    meta = train_dataset.meta

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
        num_channels=meta['num_cam_channels'],
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
        num_channels=meta['num_rad_channels'],
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

    # training parameters (num_vfe_features is actually also a training parameter)
    batch_size = cfg.batch_size 
    depth = cfg.depth
    num_heads = cfg.num_heads
    lr = cfg.lr

    train_loader = DataLoader(
        train_dataset, 
        batch_size, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_radar)

    model = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=dim_tokens,
        depth=depth,
        num_heads=num_heads
    )

    model_lightning = BevMultiMAELightning(model, lr=lr, num_encoded_tokens=cfg.num_encoded_tokens)
    trainer = Trainer(
        max_epochs = cfg.max_epochs,
        min_epochs = cfg.min_epochs,
        enable_checkpointing=False
    )

    trainer.fit(model_lightning, train_loader)


    model_lightning.model.train()
    model_lightning.model.cuda()
    with torch.no_grad():
        batch = next(iter(train_loader))
        torch.manual_seed(42)  # fixed mask for eval
        batch["cam_bev"] = batch["cam_bev"].cuda()
        for k, v in batch["radar"].items():
            if isinstance(v, torch.Tensor):
                batch["radar"][k] = v.cuda()
        batch["radar_target"] = batch["radar_target"].cuda()
        
        preds, task_masks = model_lightning.model(batch, mask_inputs=True, num_encoded_tokens=cfg.num_encoded_tokens)
        

        rad_pred = preds["radar"]
        rad_target = batch["radar_target"].cuda()
        
        # print("pred shape:", rad_pred.shape)
        # print("target shape:", rad_target.shape)
        # print("pred occ raw min/max/mean:", rad_pred[0,0].min().item(), rad_pred[0,0].max().item(), rad_pred[0,0].mean().item())
        # print("target occ sum:", rad_target[0,0].sum().item())
        # print("mse:", F.mse_loss(rad_pred, rad_target).item())
        
        # composite: pred where masked, original where visible
        B = task_masks["cam_bev"].shape[0]
        ny, nx = grid_size[1], grid_size[0]
        cam_mask = task_masks["cam_bev"].reshape(B, ny, nx)
        cam_mask = cam_mask.repeat_interleave(15, dim=1).repeat_interleave(15, dim=2).unsqueeze(1).float()
        composite = preds["cam_bev"] * cam_mask + batch["cam_bev"] * (1 - cam_mask)
        
        viz_preds({"cam_bev": composite, "radar": preds["radar"]}, batch, cfg.plot_folder)

if __name__ == '__main__':
    main()