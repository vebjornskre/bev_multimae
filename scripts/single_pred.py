import os
import torch
from torch.utils.data import DataLoader
from einops import rearrange
import logging

import hydra
from omegaconf import DictConfig

from bev_multimae.datasets.data import BEVDataset, collate_radar
from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.decoders.recon_decoder import SpatialOutputAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.multimae.model_lightning import BevMultiMAELightning
from bev_multimae.visualization.predictions import viz_preds
from bev_multimae.multimae.train_utils import *


log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sample_idx = 40
    # sample_idx = 90
    # sample_idx = 140
    # sample_idx = 190
    # sample_idx = 230
    # sample_idx = 350  
    # sample_idx = 400
    # sample_idx = 520
    # sample_idx = 580
    # sample_idx = 630
    sample_idx = 690
    # sample_idx = 740
    

    log.info(f'Predicting idx {sample_idx} in the validation set')

    try: 
        ms = torch.load(os.path.join(cfg.processed_data_dir, 'mean_std.pt'))
        img_mean, img_std, rad_mean, rad_std = (ms['img_mean'], ms['img_std'], ms['rad_mean'], ms['rad_mean'])
    except:
        img_mean, img_std = compute_img_stats(cfg.processed_data_dir) 
        rad_mean, rad_std = compute_radar_stats(cfg.processed_data_dir)
        ms = {
            'img_mean' : img_mean,
            'img_std'  : img_std,
            'rad_mean' : rad_mean,
            'rad_std'  : rad_std,
        }
        torch.save(ms, os.path.join(cfg.processed_data_dir, 'mean_std.pt'))

    ds = BEVDataset(
        cfg.processed_data_dir, split="val", 
        img_mean=img_mean, img_std=img_std,
        rad_mean=rad_mean, rad_std=rad_std,
        augment=cfg.augment, h_flip_rate=1.0,
        v_flip_rate=1.0, rot_rate=1.0,
        rot_angle=cfg.rot_angle, point_cloud_range=cfg.point_cloud_range
        )

    print(len(ds))

    sample = ds[sample_idx]
    batch = collate_radar([sample])

    meta = ds.meta
    grid_size = meta['grid_size']
    grid_size_hires = meta['hi_res_grid_size']

    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]
    patch_size = (ny_hi // ny, nx_hi // nx)

    ckpt_path = os.path.join(cfg.model_folder, f'{cfg.best_model}.ckpt')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]

    dim_tokens = cfg.dim_tokens

    input_adapters = {
        "radar": RadarAdapter(hp['dim_tokens'], grid_size, meta['num_point_features'], cfg.num_vfe_features),
        "cam_bev": CameraAdapter(hp['dim_tokens'], cfg.cam_channels, patch_size, grid_size_hires),
    }

    output_adapters = {
        "cam_bev": SpatialOutputAdapter(
            num_channels=meta['num_cam_channels'],
            stride_level=1,
            patch_size_full=patch_size,
            image_size=(grid_size_hires[1], grid_size_hires[0]),
            task="cam_bev",
            context_tasks=["cam_bev", "radar"],
            dim_tokens=hp['dim_tokens'],
            dim_tokens_enc=hp['dim_tokens'],
        ),
        "radar": SpatialOutputAdapter(
            num_channels=meta['num_rad_channels'],
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
        ckpt_path, model=model
    )

    model = model_lightning.model.to(device).eval()

    batch["cam_bev"] = batch["cam_bev"].to(device)
    for k, v in batch["radar"].items():
        if isinstance(v, torch.Tensor):
            batch["radar"][k] = v.to(device)
    batch["radar_target"] = batch["radar_target"].to(device)

    with torch.no_grad():
        preds, masks = model(
            batch,
            mask_inputs=True,
            # num_encoded_tokens=hp['num_encoded_tokens']
            num_encoded_tokens=350
        )

    img_mean = ds.img_mean.to(device)
    img_std = ds.img_std.to(device)

    cam_pred = preds["cam_bev"]

    if cfg.norm_pix:
        cam_pred = denorm_patches(cam_pred, batch["cam_bev"], patch_size[0])

    cam_pred  = denorm_img(cam_pred, img_mean, img_std)
    cam_input = denorm_img(batch["cam_bev"], img_mean, img_std)

    B = 1
    ph, pw = patch_size
    cam_mask = masks["cam_bev"].reshape(B, grid_size[1], grid_size[0])
    cam_mask = cam_mask.repeat_interleave(ph, dim=1)\
                       .repeat_interleave(pw, dim=2)\
                       .unsqueeze(1).float()

    composite = cam_pred * cam_mask + cam_input * (1 - cam_mask)

    preds["cam_bev"] = composite
    batch["cam_bev"] = cam_input

    viz_preds(preds, batch, cfg.plot_folder)


if __name__ == "__main__":
    main()