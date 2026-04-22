from pytorch_lightning import Trainer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from functools import partial
from einops import rearrange

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

def denorm_patches(pred, target, patch_size):
    """Denormalize patch-normalized predictions using target patch stats."""
    p = patch_size
    H, W = target.shape[-2:]
    nh, nw = H // p, W // p

    # patchify target to get per-patch stats
    t = rearrange(target, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
    mean = t.mean(dim=-1, keepdim=True)
    var  = t.var(dim=-1, keepdim=True)

    # patchify pred, denorm, unpatchify
    p_pred = rearrange(pred, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
    p_pred = p_pred * torch.sqrt(var + 1e-6) + mean

    return rearrange(p_pred, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p, c=target.shape[1])

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

    # training parameters (num_vfe_features is also a training parameter)
    batch_size = cfg.batch_size 
    depth = cfg.depth
    num_heads = cfg.num_heads
    lr = cfg.lr

    train_loader = DataLoader(
        train_dataset, 
        batch_size, 
        num_workers=6,
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

    ckpt_path = os.path.join(cfg.model_folder, "bev_multimae.ckpt")
    continue_training = True

    os.makedirs(cfg.model_folder, exist_ok=True)
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        enable_checkpointing=False
    )

    if continue_training and os.path.exists(ckpt_path):
        model_lightning = BevMultiMAELightning.load_from_checkpoint(
            ckpt_path, model=model, lr=lr, num_encoded_tokens=cfg.num_encoded_tokens
        )
        print(f"Loaded checkpoint from {ckpt_path}")
        trainer.fit(model_lightning, train_loader)
    else:
        trainer.fit(model_lightning, train_loader)

    trainer.save_checkpoint(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

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
            mask_inputs=True,   # set False if you want no masking
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

        print("task_masks cam_bev shape:", task_masks["cam_bev"].shape)
        print("task_masks cam_bev sum:", task_masks["cam_bev"].sum().item())
        print("cam_mask sum:", cam_mask.sum().item())

        # composite logic
        if cam_mask.sum() == 0:
            composite = preds["cam_bev"]  # no copying from input
        else:
            composite = preds["cam_bev"] * cam_mask + batch["cam_bev"] * (1 - cam_mask)

        cam_pred = denorm_patches(preds["cam_bev"], batch["cam_bev"], patch_size=15)
        composite = cam_pred * cam_mask + batch["cam_bev"] * (1 - cam_mask)

        viz_preds(
            {"cam_bev": composite, "radar": preds["radar"]},
            batch,
            cfg.plot_folder
        )

        print("preds cam_bev:", preds["cam_bev"].shape)
        print("cam_mask:", cam_mask.shape)
        print("batch cam_bev:", batch["cam_bev"].shape)

if __name__ == '__main__':
    main()