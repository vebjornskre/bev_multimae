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
from bev_multimae.multimae.train_utils import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_results(mode, images: dict, metrics: dict):

    save_dir = "reports/figures/diagnostic_plots"
    os.makedirs(save_dir, exist_ok=True)

    titles = list(images.keys())
    n = len(titles)
    has_diff = n == 2

    cols = n + (1 if has_diff else 0)
    fig = plt.figure(figsize=(4 * cols, 4.5))
    fig.patch.set_facecolor("#0f0f0f")
    gs = gridspec.GridSpec(1, cols, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(cols)]

    def to_np(t):
        t = t.squeeze(0).detach().cpu().float()
        if t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        t = t.clamp(0, 1).numpy()
        return t

    imgs_np = [to_np(v) for v in images.values()]

    for ax, img, title in zip(axes[:n], imgs_np, titles):
        ax.imshow(img if img.shape[-1] == 3 else img.squeeze(-1), cmap="inferno")
        ax.set_title(title, color="#e0e0e0", fontsize=9, pad=6)
        ax.axis("off")

    if has_diff:
        diff_map = np.abs(imgs_np[0].astype(float) - imgs_np[1].astype(float)).mean(-1)
        im = axes[n].imshow(diff_map, cmap="hot", vmin=0, vmax=1)
        axes[n].set_title("diff", color="#ff6b6b", fontsize=9, pad=6)
        axes[n].axis("off")
        plt.colorbar(im, ax=axes[n], fraction=0.046, pad=0.04)

    metric_str = "  |  ".join([f"{k}: {v:.5f}" for k, v in metrics.items()])
    fig.suptitle(f"[{mode}]  {metric_str}", color="#ffffff", fontsize=10, y=1.01, fontweight="bold")
    plt.tight_layout()

    fname = f"{mode}_{'_'.join([f'{k[:6]}{v:.4f}' for k,v in metrics.items()])}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_diagnostic(model, ds, collate_radar, device, grid_size, patch_size,
                   cfg, ds_right, mode="values_vs_structure"):
    """
    Diagnostic modes:
      - radar_shuffle          : shuffle radar tokens spatially
      - mean_radar             : replace radar floats with channel mean
      - noise_radar            : replace radar floats with gaussian noise
      - zero_radar             : zero all radar floats
      - zero_camera            : zero camera input
      - zero_both              : zero camera + zero radar
      - determinism            : same zero input twice, check outputs match
      - cross_sample           : two different samples, real radar
      - force_identical_radar  : two samples, forced same radar, radar only
      - values_vs_structure    : same structure, randomized float values
      - cam_patch_probe        : identical radar, one camera patch through at (row, col)
    """

    SEED = 0
    IDX_A, IDX_B = 40, 150

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    def to_device(batch):
        batch["cam_bev"] = batch["cam_bev"].to(device)
        batch["radar_target"] = batch["radar_target"].to(device)
        for k, v in batch["radar"].items():
            if isinstance(v, torch.Tensor):
                batch["radar"][k] = v.to(device)
        return batch

    def clone_radar(src):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in src["radar"].items()}

    def infer(batch, cam_mask=None):
        B = 1
        N = grid_size[0] * grid_size[1]

        if cam_mask is None:
            cam_mask = torch.ones(B, N, device=device)

        n_cam_unmasked = int((cam_mask == 0).sum().item())

        task_masks = {
            "cam_bev": cam_mask,
            "radar": torch.zeros(B, N, device=device),
        }

        preds, _ = model(
            batch,
            mask_inputs=True,
            task_masks=task_masks,
            num_encoded_tokens=N + n_cam_unmasked
        )

        img_mean = ds_right.img_mean.to(device)
        img_std = ds_right.img_std.to(device)

        cam_pred = preds["cam_bev"]

        pred = denorm_img(cam_pred, img_mean, img_std)

        return pred, cam_mask

    def make_composite(pred, cam_input, cam_mask):
        B = 1
        ph, pw = patch_size
        nx, ny = grid_size[:2]
        mask_2d = cam_mask.reshape(B, ny, nx)
        mask_2d = mask_2d.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2).unsqueeze(1).float()
        return pred * mask_2d + cam_input * (1 - mask_2d)

    def make_cam_mask(positions):
        # positions: list of (row, col) tuples
        N = grid_size[0] * grid_size[1]
        nw = grid_size[0]
        mask = torch.ones(1, N, device=device)
        for row, col in positions:
            mask[:, row * nw + col] = 0
        return mask

    def diff(a, b):
        return torch.abs(a - b).mean().item()

    with torch.no_grad():
        a = to_device(collate_radar([ds[IDX_A]]))
        b = to_device(collate_radar([ds[IDX_B]]))

        img_mean = ds_right.img_mean.to(device)
        img_std  = ds_right.img_std.to(device)

        cam_a = denorm_img(a["cam_bev"], img_mean, img_std)
        cam_b = denorm_img(b["cam_bev"], img_mean, img_std)

        plot_results(
            "original_inputs",
            {"IDX_A_40": cam_a, "IDX_B_150": cam_b},
            {"dummy": 0.0}
        )

        if mode == "radar_shuffle":
            for k, v in a["radar"].items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    a["radar"][k] = v[:, torch.randperm(v.shape[1], device=device)]
            pred_orig, _ = infer(b)
            pred_shuf, _ = infer(a)
            d = diff(pred_orig, pred_shuf)
            print(f"[radar_shuffle] diff: {d:.6f}")
            plot_results(mode, {"original": pred_orig, "shuffled": pred_shuf}, {"mean abs diff": d})

        elif mode == "mean_radar":
            for k, v in a["radar"].items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    a["radar"][k] = v.mean(dim=1, keepdim=True).expand_as(v)
            pred, _ = infer(a)
            print(f"[mean_radar] output mean: {pred.mean().item():.6f}")
            plot_results(mode, {"mean radar pred": pred}, {"output mean": pred.mean().item()})

        elif mode == "noise_radar":
            for k, v in a["radar"].items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    a["radar"][k] = torch.randn_like(v)
            pred, _ = infer(a)
            print(f"[noise_radar] output mean: {pred.mean().item():.6f}")
            plot_results(mode, {"noise radar pred": pred}, {"output mean": pred.mean().item()})

        elif mode == "zero_radar":
            for k, v in a["radar"].items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    a["radar"][k] = torch.zeros_like(v)
            pred, _ = infer(a)
            print(f"[zero_radar] output mean: {pred.mean().item():.6f}")
            plot_results(mode, {"zero radar pred": pred}, {"output mean": pred.mean().item()})

        elif mode == "zero_camera":
            a["cam_bev"] = torch.zeros_like(a["cam_bev"])
            pred, _ = infer(a)
            print(f"[zero_camera] output mean: {pred.mean().item():.6f}")
            plot_results(mode, {"zero cam pred": pred}, {"output mean": pred.mean().item()})

        elif mode == "zero_both":
            a["cam_bev"] = torch.zeros_like(a["cam_bev"])
            for k, v in a["radar"].items():
                if isinstance(v, torch.Tensor):
                    a["radar"][k] = torch.zeros_like(v) if v.dtype.is_floating_point else v
            pred, _ = infer(a)
            print(f"[zero_both] output mean: {pred.mean().item():.6f}")
            plot_results(mode, {"zero both pred": pred}, {"output mean": pred.mean().item()})

        elif mode == "determinism":
            a["cam_bev"] = torch.zeros_like(a["cam_bev"])
            for k, v in a["radar"].items():
                if isinstance(v, torch.Tensor):
                    a["radar"][k] = torch.zeros_like(v) if v.dtype.is_floating_point else v
            pred_1, _ = infer(a)
            pred_2, _ = infer(a)
            d = diff(pred_1, pred_2)
            print(f"[determinism] diff between runs: {d:.8f}")
            plot_results(mode, {"run 1": pred_1, "run 2": pred_2}, {"diff between runs": d})

        elif mode == "cross_sample":
            pred_a, _ = infer(a)
            pred_b, _ = infer(b)
            d = diff(pred_a, pred_b)
            print(f"[cross_sample] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": pred_a, "sample B": pred_b}, {"mean abs diff": d})

        elif mode == "force_identical_radar":
            fixed = clone_radar(a)
            a["radar"] = fixed
            b["radar"] = clone_radar(a)

            pred_a, _ = infer(a)
            pred_b, _ = infer(b)
            d = diff(pred_a, pred_b)
            print(f"[force_identical_radar] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": pred_a, "sample B": pred_b}, {"mean abs diff": d})

        elif mode == "values_vs_structure":
            fixed = clone_radar(a)
            a["radar"] = fixed
            b["radar"] = clone_radar(a)
            a["cam_bev"] = torch.zeros_like(a["cam_bev"])
            b["cam_bev"] = torch.zeros_like(b["cam_bev"])
            for k, v in b["radar"].items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    b["radar"][k] = torch.randn_like(v)
            pred_a, _ = infer(a)
            pred_b, _ = infer(b)
            d = diff(pred_a, pred_b)
            print(f"[values_vs_structure] diff: {d:.6f}")
            plot_results(mode, {"original values": pred_a, "random values": pred_b}, {"mean abs diff": d})

        elif mode == "cam_patch_probe":
            fixed = clone_radar(a)
            a["radar"] = fixed
            b["radar"] = clone_radar(a)

            # positions = [
            #     (5, 8), (6, 9), (4, 6),
            #     (7, 10), (3, 7), (8, 5),
            #     (2, 12), (9, 3), (6, 11),
            #     (4, 9)
            # ]

            positions = [
                (5, 8), (6, 9), (4, 6), (7, 10), (3, 7), (8, 5), (2, 12), (9, 3), (6, 11), (4, 9),
                (1, 4), (10, 6), (11, 8), (12, 2), (0, 7), (13, 9), (14, 5), (15, 10), (2, 3), (7, 1),
                (16, 16), (17, 0), (0, 17), (17, 17), (16, 1), (1, 16), (12, 15), (15, 12), (9, 17), (17, 9),
                (0, 0), (0, 5), (0, 10), (0, 15),
                (5, 0), (5, 5), (5, 10), (5, 15),
                (10, 0), (10, 5), (10, 10), (10, 15),
                (15, 0), (15, 5), (15, 10), (15, 15),
                (8, 8), (9, 9), (8, 9), (9, 8),
                (3, 14), (14, 3), (6, 16), (16, 6)
            ]
            cam_mask = make_cam_mask(positions)

            img_mean = ds_right.img_mean.to(device)
            img_std = ds_right.img_std.to(device)
            cam_input_a = denorm_img(a["cam_bev"], img_mean, img_std)
            cam_input_b = denorm_img(b["cam_bev"], img_mean, img_std)

            pred_a, mask = infer(a, cam_mask=cam_mask)
            pred_b, _    = infer(b, cam_mask=cam_mask)

            comp_a = make_composite(pred_a, cam_input_a, mask)
            comp_b = make_composite(pred_b, cam_input_b, mask)

            d = diff(comp_a, comp_b)
            print(f"[cam_patch_probe] positions={positions} diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": comp_a, "sample B": comp_b}, {"mean abs diff": d})

        else:
            raise ValueError(f"Unknown mode: {mode}")

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_idx = 1200

    log.info(f'Predicting idx {sample_idx} in the validation set')

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
    batch = collate_radar([sample])

    meta = ds_right.meta
    grid_size = meta['grid_size']
    grid_size_hires = meta['hi_res_grid_size']

    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]
    patch_size = (ny_hi // ny, nx_hi // nx)

    ckpt_path = os.path.join(cfg.model_folder, f'{cfg.best_model}.ckpt')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]

    dim_tokens = cfg.dim_tokens
    hp['num_cam_channels'] = 11

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
            num_channels=hp['num_rad_channels'],
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

    run_diagnostic(model, ds, collate_radar, device, grid_size, patch_size, cfg, ds_right, mode="force_identical_radar")

    #### Old code - uses both modalities like in training ####

    # with torch.no_grad():
    #     preds, masks = model(
    #         batch,
    #         mask_inputs=True,
    #         num_encoded_tokens=158
    #     )

    # img_mean = ds_right.img_mean.to(device)
    # img_std  = ds_right.img_std.to(device)

    # cam_pred  = preds["cam_bev"]
    # cam_input = batch["cam_bev"]

    # B = cam_pred.shape[0]
    # ph, pw = patch_size

    # cam_mask = masks["cam_bev"].reshape(B, grid_size[1], grid_size[0])
    # cam_mask = cam_mask.repeat_interleave(ph, dim=1)\
    #                 .repeat_interleave(pw, dim=2)\
    #                 .unsqueeze(1).float()

    # cam_composite = cam_pred * cam_mask + cam_input * (1 - cam_mask)

    # cam_composite = denorm_img(cam_composite, img_mean, img_std)

    # # Radar
    # rad_pred = preds["radar"]
    # rad_input = batch["radar_target"]

    # rad_mask = masks["radar"].reshape(B, grid_size[1], grid_size[0])\
    #                         .unsqueeze(1).float()

    # rad_composite = rad_pred * rad_mask + rad_input * (1 - rad_mask)

    # preds["cam_bev"] = cam_composite
    # preds["radar"] = rad_composite

    # batch["cam_bev"] = denorm_img(cam_input, img_mean, img_std)

    # viz_preds(preds, batch, cfg.plot_folder)


if __name__ == "__main__":
    main()