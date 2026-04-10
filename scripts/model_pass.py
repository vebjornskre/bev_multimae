import os
import torch
import torch.nn as nn
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
from bev_multimae.datasets.data import BEVDataset, collate_radar

import matplotlib.pyplot as plt

def save_preds(preds, folder):

    for k, v in preds.items():
        x = v[0].detach().cpu()

        if x.dim() == 3:
            x = x.permute(1, 2, 0)

        x = x.numpy()
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        plt.figure()

        if k == "radar":
            plt.imshow(x[..., 0], cmap='gray')
        else:
            plt.imshow(x[..., :3])

        plt.axis('off')
        plt.savefig(f"{folder}/{k}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_path = cfg.processed_data_dir
    train_dataset = BEVDataset(data_path, cfg)
    meta = train_dataset.meta

    grid_size = meta['grid_size']
    grid_size_hires = meta['hi_res_grid_size']

    H_cam, W_cam = grid_size_hires[:2]
    patch_h = H_cam // grid_size[0]
    patch_w = W_cam // grid_size[1]
    patch_size = (patch_h, patch_w)

    num_point_features = meta['num_point_features']
    num_vfe_features = cfg.num_vfe_features

    dim_tokens = cfg.dim_tokens # embed_dim

    rad_adapt = RadarAdapter(dim_tokens, grid_size, num_point_features, num_vfe_features)
    cam_adapt = CameraAdapter(dim_tokens, cfg.cam_channels, patch_size, grid_size_hires)

    # training parameters
    batch_size = cfg.batch_size 

    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_radar)

    batch = next(iter(train_loader))


    # rad_out = rad_adapt(batch["radar"])
    # cam_out = cam_adapt(batch["cam_bev"])

    input_adapters = {
        'radar' : rad_adapt,
        'cam_bev'   : cam_adapt
    }

    cam_decode = SpatialOutputAdapter(
        num_channels=meta['num_cam_channels'],
        stride_level=1,          # reconstructing at full hires resolution
        patch_size_full=patch_size,
        image_size=grid_size_hires,
        task='cam_bev',
        context_tasks=['cam_bev', 'radar'],
        dim_tokens_enc=dim_tokens
    )
    rad_decode = SpatialOutputAdapter(
        num_channels=meta['num_rad_channels'],
        stride_level=1,
        patch_size_full=(1, 1),  # each token is already one grid cell
        image_size=grid_size,
        task='radar',
        context_tasks=['cam_bev', 'radar'],
        dim_tokens_enc=dim_tokens
    )

    output_adapters = {
        'cam_bev': cam_decode,
        'radar': rad_decode,
    }

    model = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=dim_tokens,
        depth=12,
        num_heads=8
    )

    preds, _ = model(batch)
    save_preds(preds, cfg.plot_folder)

    

if __name__ == '__main__':
    main()