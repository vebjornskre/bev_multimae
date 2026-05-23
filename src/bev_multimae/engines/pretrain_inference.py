# This script will contain inference code
import os
import torch
from torch.utils.data import DataLoader
from einops import rearrange
import logging
import torch.nn as nn

import hydra
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset

from bev_multimae.datasets.data import BEVDataset, collate_radar
from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.decoders.recon_decoder import SpatialOutputAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.multimae.model_lightning import BevMultiMAELightning
from bev_multimae.visualization.predictions import viz_preds
from bev_multimae.engines.train_utils import *
from bev_multimae.engines.inference_utils import run_diagnostic

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

log = logging.getLogger(__name__)



def to_device(batch, device):
    batch["cam_bev"] = batch["cam_bev"].to(device)
    batch["radar_target"] = batch["radar_target"].to(device)
    for k, v in batch["radar"].items():
        if isinstance(v, torch.Tensor):
            batch["radar"][k] = v.to(device)
    return batch

def setup_data(cfg, sample_idx, device):
    try:
        ms = torch.load(os.path.join(cfg.processed_data_dir, 'mean_std.pt'))
        img_mean, img_std, rad_mean, rad_std = (ms['img_mean'], ms['img_std'], ms['rad_mean'], ms['rad_std'])
    except:
        img_mean, img_std = compute_img_stats(cfg.processed_data_dir)
        rad_mean, rad_std = compute_radar_stats(cfg.processed_data_dir)
        ms = {
            'img_mean': img_mean,
            'img_std':  img_std,
            'rad_mean': rad_mean,
            'rad_std':  rad_std,
        }
        torch.save(ms, os.path.join(cfg.processed_data_dir, 'mean_std.pt'))

    ds_right = BEVDataset(
        cfg.processed_data_dir_right, split="val",
        img_mean=img_mean, img_std=img_std,
        rad_mean=rad_mean, rad_std=rad_std,
        augment=cfg.augment, h_flip_rate=0.0,
        v_flip_rate=0.0, rot_rate=0.0,
        rot_angle=cfg.rot_angle, point_cloud_range=cfg.right_point_cloud_range
    )
    ds_left = BEVDataset(
        cfg.processed_data_dir_left, split="val",
        img_mean=img_mean, img_std=img_std,
        rad_mean=rad_mean, rad_std=rad_std,
        augment=cfg.augment, h_flip_rate=0.0,
        v_flip_rate=0.0, rot_rate=0.0,
        rot_angle=cfg.rot_angle, point_cloud_range=cfg.left_point_cloud_range
    )

    ds = ConcatDataset([ds_right, ds_left])
    sample = ds[sample_idx]
    sample_batch = collate_radar([sample])

    sample_batch = to_device(sample_batch, device)

    return ds, sample_batch

def setup_model(cfg, meta, grid_size, grid_size_hires, patch_size, device):

    ckpt_path = os.path.join(cfg.model_folder, f'{cfg.best_model}.ckpt')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]

    input_adapters = {
        "radar": RadarAdapter(hp['dim_tokens'], grid_size, meta['num_point_features'], cfg.num_vfe_features),
        "cam_bev": CameraAdapter(hp['dim_tokens'], cfg.cam_channels, patch_size, grid_size_hires),
    }

    cam_channels = hp.get("num_cam_channels", cfg.cam_channels)
    rad_channels = hp.get("num_rad_channels", cfg.rad_channels)

    output_adapters = {
        "cam_bev": SpatialOutputAdapter(
            num_channels=cam_channels,
            stride_level=1,
            patch_size_full=patch_size,
            image_size=(grid_size_hires[1], grid_size_hires[0]),
            task="cam_bev",
            context_tasks=["cam_bev", "radar"],
            dim_tokens=hp['dim_tokens'],
            dim_tokens_enc=hp['dim_tokens'],
        ),
        "radar": SpatialOutputAdapter(
            num_channels=rad_channels,
            stride_level=1,
            patch_size_full=(1, 1),
            image_size=(grid_size[1], grid_size[0]),
            task="radar",
            context_tasks=["cam_bev", "radar"],
            dim_tokens_enc=hp['dim_tokens'],
        ),
    }

    model = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=hp["dim_tokens"],
        depth=hp["depth"],
        num_heads=hp["num_heads"],
    )

    model_lightning = BevMultiMAELightning.load_from_checkpoint(
        ckpt_path, model=model, strict=False
    )

    model = model_lightning.model.to(device).eval()

    return model

def infer(model, sample_batch, num_encoded_tokens=158):

    with torch.no_grad():
        preds, masks = model(
            sample_batch,
            mask_inputs=True,
            num_encoded_tokens=num_encoded_tokens
        )

    return preds, masks

def compose_norm(preds, sample_batch, masks, img_mean, img_std, grid_size, patch_size):

    cam_pred  = preds["cam_bev"]
    cam_input = sample_batch["cam_bev"]

    B = cam_pred.shape[0]
    ph, pw = patch_size

    cam_mask = masks["cam_bev"].reshape(B, grid_size[1], grid_size[0])
    cam_mask = cam_mask.repeat_interleave(ph, dim=1)\
                    .repeat_interleave(pw, dim=2)\
                    .unsqueeze(1).float()

    cam_composite = cam_pred * cam_mask + cam_input * (1 - cam_mask)

    cam_composite = denorm_img(cam_composite, img_mean, img_std)

    # Radar
    rad_pred = preds["radar"]
    rad_input = sample_batch["radar_target"]

    rad_mask = masks["radar"].reshape(B, grid_size[1], grid_size[0])\
                            .unsqueeze(1).float()

    rad_composite = rad_pred * rad_mask + rad_input * (1 - rad_mask)

    preds["cam_bev"] = cam_composite
    preds["radar"] = rad_composite

    sample_batch["cam_bev"] = denorm_img(cam_input, img_mean, img_std)

    return preds, sample_batch


def setup_and_infer(cfg: DictConfig, sample_idx, visualize=True, diagnostic=False, diag_mode=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, sample_batch = setup_data(cfg, sample_idx, device)

    meta = ds.datasets[0].meta
    grid_size = meta['grid_size']
    grid_size_hires = meta['hi_res_grid_size']

    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]
    patch_size = (ny_hi // ny, nx_hi // nx)

    model = setup_model(cfg, meta, grid_size, grid_size_hires, patch_size, device)

    preds, masks = infer(model, sample_batch)
    img_mean = ds.datasets[0].img_mean.to(device)
    img_std  = ds.datasets[0].img_std.to(device)

    composite, sample_batch = compose_norm(
        preds, sample_batch, 
        masks, img_mean, img_std, 
        grid_size, patch_size
        )

    if diagnostic:
        run_diagnostic(model, ds, collate_radar, device, grid_size, patch_size, cfg, mode=diag_mode)

    if visualize:
        viz_preds(composite, sample_batch, cfg.plot_folder)

    return composite, sample_batch

def main():
    raise RuntimeError('Script should not be run alone. Try: \n uv run scripts/single_pred.py')

if __name__ == "__main__":
    main()