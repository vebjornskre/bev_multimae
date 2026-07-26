import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random

try:
    from bev_multimae.datasets.data_with_feat import collate_radar
except ImportError:
    from bev_multimae.datasets.data import collate_radar
from bev_multimae.engines.train_utils import *



def plot_results(mode, images: dict, metrics: dict, batches=None, point_cloud_range=None):

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

    for i, (ax, img, title) in enumerate(zip(axes[:n], imgs_np, titles)):
        ax.imshow(img if img.shape[-1] == 3 else img.squeeze(-1), cmap="inferno")
        ax.set_title(title, color="#e0e0e0", fontsize=9, pad=6)

        if batches is not None and point_cloud_range is not None:
            batch = batches[i]

            pts = batch["radar"]["points"].detach().cpu().numpy()
            pts = pts[pts[:, 0] == 0]

            x = pts[:, 1]
            y = pts[:, 2]

            x_min, y_min, _, x_max, y_max, _ = point_cloud_range
            H, W = img.shape[:2]

            px = (x - x_min) / (x_max - x_min) * (W - 1)
            py = (y - y_min) / (y_max - y_min) * (H - 1)

            inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            ax.scatter(px[inside], py[inside], s=2, c="lime", alpha=0.7)

        ax.axis("off")

    if has_diff:
        diff_map = np.abs(imgs_np[0].astype(float) - imgs_np[1].astype(float)).mean(-1)
        im = axes[n].imshow(diff_map, cmap="hot", vmin=0, vmax=1)
        axes[n].set_title("Difference", color="#ff6b6b", fontsize=9, pad=6)
        axes[n].axis("off")
        plt.colorbar(im, ax=axes[n], fraction=0.046, pad=0.04)

    metric_str = "  |  ".join([f"{k}: {v:.5f}" for k, v in metrics.items()])
    fig.suptitle(f"[{mode}]  {metric_str}", color="#ffffff", fontsize=10, y=1.01, fontweight="bold")
    plt.tight_layout()

    fname = f"{mode}_{'_'.join([f'{k[:6]}{v:.4f}' for k,v in metrics.items()])}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_radar_results(mode, pred, target, mask, metrics, step, pred_occ_reg=False):
    save_dir = f"reports/figures/diagnostic_plots/{mode}/visible_{step:03d}"
    os.makedirs(save_dir, exist_ok=True)

    rad_titles = [
        "Occupancy", "Density (log count)", "Height (mean z)", "Velocity (mean)", "RCS (mean)",
        "Height (var)", "Velocity (var)", "RCS (var)", "SNR (mean)",
        "Mean x from pillar center", "Mean y from pillar center",
    ]

    def safe_percentile(x, q, default=1.0):
        x = np.asarray(x)
        if np.all(np.isnan(x)):
            return default
        return np.nanpercentile(x, q)

    pred = pred.squeeze(0).detach().cpu().float()
    target = target.squeeze(0).detach().cpu().float()
    mask = mask.squeeze(0).squeeze(0).detach().cpu().float()

    occ = torch.sigmoid(pred[0]) > 0.5
    occ_target = target[0] > 0.5

    pred = pred.clone()
    target = target.clone()
    pred[0] = occ.float()

    reg_mask = occ if pred_occ_reg else occ_target

    for ch in range(1, pred.shape[0]):
        pred[ch] = torch.where(reg_mask, pred[ch], torch.nan)
        target[ch] = torch.where(occ_target, target[ch], torch.nan)

    metric_str = "  |  ".join(f"{k}: {v:.5f}" for k, v in metrics.items())

    for ch in range(pred.shape[0]):
        title = rad_titles[ch] if ch < len(rad_titles) else f"Channel {ch}"
        imgs = [pred[ch].numpy(), target[ch].numpy(), mask.numpy()]
        names = [f"Prediction: {title}", f"Target: {title}", "visible mask"]

        if ch == 0:
            cmap, vmin, vmax = "gray", 0, 1
        elif any(s in title.lower() for s in ["var", "density", "snr"]):
            cmap = "viridis"
            vmin = safe_percentile(imgs[1], 2, default=0.0)
            vmax = safe_percentile(imgs[1], 98, default=1.0)
        else:
            lim = safe_percentile(np.abs(imgs[1]), 98, default=1.0)
            cmap, vmin, vmax = "coolwarm", -lim, lim

        if vmin == vmax:
            vmax = vmin + 1e-6

        main_cmap = plt.get_cmap(cmap).copy()
        main_cmap.set_bad("black")

        diff_cmap = plt.get_cmap("magma").copy()
        diff_cmap.set_bad("black")

        fig = plt.figure(figsize=(16, 4.5))
        fig.patch.set_facecolor("white")

        gs = gridspec.GridSpec(1, 4, figure=fig)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        for ax in axes:
            ax.set_facecolor("white")

        im_main = None
        for i, (ax, img, name) in enumerate(zip(axes[:3], imgs, names)):
            if i < 2:
                im_main = ax.imshow(img, cmap=main_cmap, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, cmap="bwr", vmin=0, vmax=1)

            ax.set_title(name, color="black", fontsize=9, pad=6)
            ax.axis("off")

        diff = np.abs(imgs[0] - imgs[1])
        diff_vmax = safe_percentile(diff, 98, default=1.0)
        if diff_vmax == 0:
            diff_vmax = 1e-6

        im_diff = axes[3].imshow(diff, cmap=diff_cmap, vmin=0, vmax=diff_vmax)
        axes[3].set_title("abs diff", color="black", fontsize=9, pad=6)
        axes[3].axis("off")

        cbar_main = plt.colorbar(im_main, ax=axes[:2], fraction=0.025, pad=0.02)
        cbar_main.ax.tick_params(labelsize=8, colors="black")
        cbar_main.outline.set_edgecolor("black")

        cbar_diff = plt.colorbar(im_diff, ax=axes[3], fraction=0.046, pad=0.04)
        cbar_diff.ax.tick_params(labelsize=8, colors="black")
        cbar_diff.outline.set_edgecolor("black")

        fig.suptitle(
            f"[{mode}] visible={step} | {title} | {metric_str}",
            color="black", fontsize=10, y=1.01, fontweight="bold"
        )

        safe = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(
            os.path.join(save_dir, f"ch_{ch:02d}_{safe}.png"),
            dpi=200,
            bbox_inches="tight",
            facecolor="white"
        )
        plt.close(fig)

def run_diagnostic(model, ds, collate_radar, device, grid_size, patch_size,
                   cfg, mode="values_vs_structure"):
    """
    Diagnostic modes:
      - radar_shuffle                  : shuffle radar tokens spatially
      - mean_radar                     : replace radar floats with channel mean
      - noise_radar                    : replace radar floats with gaussian noise
      - zero_radar                     : zero all radar floats
      - zero_camera                    : zero camera input
      - mask_all                       : 0% information
      - determinism                    : same zero input twice, check outputs match
      - cross_sample                   : two different samples, real radar
      - force_identical_radar          : two samples, forced same radar, radar only
      - values_vs_structure            : same structure, randomized float values
      - cam_patch_probe_radar          : introduce cam patches with radar only (no feat)
      - cam_patch_probe_feat           : introduce cam patches with feat only (no radar)
      - rad_patch_probe_radar          : introduce radar patches with no feat, predict radar
      - rad_patch_probe_feat           : introduce radar patches with feat visible, predict radar
      - rad_patch_probe_with_target_radar : incremental radar patches, no feat
      - rad_patch_probe_with_target_feat  : incremental radar patches, feat visible
      - feat_patch_probe_radar         : introduce feat patches with radar only (no cam)
      - feat_patch_probe_cam           : introduce feat patches with cam only (no radar)
      - rad_from_camera                : predict radar from camera with chosen number of visible radar patches
    """

    SEED = 0
    # CAMERA ABLATION
    # IDX_A, IDX_B = 349, 150
    # IDX_A, IDX_B = 150, 40
    # IDX_A, IDX_B = 299, 40
    # IDX_A, IDX_B = 400, 40
    # IDX_A, IDX_B = 418, 40 # good for ablation study

    # RADAR ABLATION
    # IDX_A, IDX_B = 12, 40 # Good candidate
    # IDX_A, IDX_B = 14, 40 # keep

    # IDX_A, IDX_B = 221, 40 

    IDX_A, IDX_B = 46, 159
    IDX_A, IDX_B = 2, 59

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    def to_device(batch):
        batch["cam_bev"] = batch["cam_bev"].to(device)
        batch["radar_target"] = batch["radar_target"].to(device)
        if "bev_feat" in batch:
            batch["bev_feat"] = batch["bev_feat"].to(device)
        for k, v in batch["radar"].items():
            if isinstance(v, torch.Tensor):
                batch["radar"][k] = v.to(device)
        return batch

    def clone_radar(src):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in src["radar"].items()}

    def zero_radar(batch):
        for k, v in batch["radar"].items():
            if isinstance(v, torch.Tensor):
                batch["radar"][k] = torch.zeros_like(v)

    def infer(batch, cam_mask=None, use_radar=True, use_feat=True):
        B = 1
        N = grid_size[0] * grid_size[1]

        if cam_mask is None:
            cam_mask = torch.ones(B, N, device=device)

        n_cam_unmasked = int((cam_mask == 0).sum().item())

        task_masks = {
            "cam_bev": cam_mask,
            "radar": torch.ones(B, N, device=device) if not use_radar else torch.zeros(B, N, device=device),
        }
        if "bev_feat" in batch:
            task_masks["bev_feat"] = torch.ones(B, N, device=device) if not use_feat else torch.zeros(B, N, device=device)

        preds, _ = model(
            batch,
            mask_inputs=True,
            task_masks=task_masks,
            num_encoded_tokens=N + n_cam_unmasked
        )

        img_mean = ds.datasets[0].img_mean.to(device)
        img_std = ds.datasets[0].img_std.to(device)

        pred = denorm_img(preds["cam_bev"], img_mean, img_std)
        return pred, cam_mask

    def infer_rad(batch, rad_mask=None, use_feat=True, ch=1):
        B = 1
        N = grid_size[0] * grid_size[1]

        if rad_mask is None:
            rad_mask = torch.ones(B, N, device=device)

        n_rad_unmasked = int((rad_mask == 0).sum().item())

        task_masks = {
            "cam_bev": torch.zeros(B, N, device=device),
            "radar": rad_mask,
        }
        if "bev_feat" in batch:
            task_masks["bev_feat"] = torch.ones(B, N, device=device) if not use_feat else torch.zeros(B, N, device=device)

        preds, _ = model(
            batch,
            mask_inputs=True,
            task_masks=task_masks,
            num_encoded_tokens=N + n_rad_unmasked
        )

        rad = preds["radar"]
        occ = (torch.sigmoid(rad[:, 0:1]) > 0.5).float()
        dens = rad[:, ch:ch + 1] * occ

        return dens, rad_mask, occ

    def infer_feat(batch, feat_mask=None, use_radar=True, use_cam=True):
        B = 1
        N = grid_size[0] * grid_size[1]

        if feat_mask is None:
            feat_mask = torch.ones(B, N, device=device)

        n_feat_unmasked = int((feat_mask == 0).sum().item())

        task_masks = {
            "cam_bev": torch.ones(B, N, device=device) if not use_cam else torch.zeros(B, N, device=device),
            "radar": torch.ones(B, N, device=device) if not use_radar else torch.zeros(B, N, device=device),
        }
        if "bev_feat" in batch:
            task_masks["bev_feat"] = feat_mask

        preds, _ = model(
            batch,
            mask_inputs=True,
            task_masks=task_masks,
            num_encoded_tokens=N + n_feat_unmasked
        )

        img_mean = ds.datasets[0].img_mean.to(device)
        img_std = ds.datasets[0].img_std.to(device)

        pred = denorm_img(preds["cam_bev"], img_mean, img_std)
        return pred, feat_mask

    def make_composite(pred, cam_input, cam_mask):
        B = 1
        ph, pw = patch_size
        nx, ny = grid_size[:2]
        mask_2d = cam_mask.reshape(B, ny, nx)
        mask_2d = mask_2d.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2).unsqueeze(1).float()
        return pred * mask_2d + cam_input * (1 - mask_2d)

    def make_rad_composite(pred, rad_input, rad_mask, occ):
        mask_2d = rad_mask.reshape(1, grid_size[1], grid_size[0]).unsqueeze(1).float()
        return pred * mask_2d + rad_input * (1 - mask_2d) * occ

    def make_cam_mask(positions):
        N = grid_size[0] * grid_size[1]
        nw = grid_size[0]
        mask = torch.ones(1, N, device=device)
        for row, col in positions:
            mask[:, row * nw + col] = 0
        return mask

    def make_rad_mask(positions):
        N = grid_size[0] * grid_size[1]
        nw = grid_size[0]
        mask = torch.ones(1, N, device=device)
        for row, col in positions:
            mask[:, row * nw + col] = 0
        return mask

    def make_feat_mask(positions):
        N = grid_size[0] * grid_size[1]
        nw = grid_size[0]
        mask = torch.ones(1, N, device=device)
        for row, col in positions:
            mask[:, row * nw + col] = 0
        return mask

    def diff(a, b):
        return torch.abs(a - b).mean().item()

    def sample_positions(n=10):
        return random.sample(
            [(row, col) for row in range(grid_size[1]) for col in range(grid_size[0])],
            n
        )

    with torch.no_grad():
        a = to_device(collate_radar([ds[IDX_A]]))
        b = to_device(collate_radar([ds[IDX_B]]))

        img_mean = ds.datasets[0].img_mean.to(device)
        img_std  = ds.datasets[0].img_std.to(device)

        cam_a = denorm_img(a["cam_bev"], img_mean, img_std)
        cam_b = denorm_img(b["cam_bev"], img_mean, img_std)

        plot_results(
            "original_inputs",
            {"IDX_A_40": cam_a, "IDX_B_150": cam_b},
            {"dummy": 0.0},
            batches=[a, b],
            point_cloud_range=[1.5, -6, -5, 28.5, 21, 20],
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

        elif mode == "mask_all":
            B = 1
            N = grid_size[0] * grid_size[1]

            task_masks = {
                "cam_bev": torch.ones(B, N, device=device),
                "radar": torch.ones(B, N, device=device),
            }

            if "bev_feat" in a:
                task_masks["bev_feat"] = torch.ones(B, N, device=device)

            preds, _ = model(
                a,
                mask_inputs=True,
                task_masks=task_masks,
                num_encoded_tokens=0,
            )

            pred = denorm_img(preds["cam_bev"], img_mean, img_std)

            print(f"[mask_all] output mean: {pred.mean().item():.6f}")
            plot_results(mode, {"fully masked pred": pred}, {"output mean": pred.mean().item()})

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

        # cam patch probes
        elif mode == "cam_patch_probe_radar":
            # introduce cam patches with radar only, no feat
            fixed = clone_radar(a)
            a["radar"] = fixed
            b["radar"] = clone_radar(a)

            cam_mask = make_cam_mask(sample_positions(40))
            cam_input_a = denorm_img(a["cam_bev"], img_mean, img_std)
            cam_input_b = denorm_img(b["cam_bev"], img_mean, img_std)

            pred_a, mask = infer(a, cam_mask=cam_mask, use_radar=True, use_feat=False)
            pred_b, _    = infer(b, cam_mask=cam_mask, use_radar=True, use_feat=False)

            comp_a = make_composite(pred_a, cam_input_a, mask)
            comp_b = make_composite(pred_b, cam_input_b, mask)

            d = diff(comp_a, comp_b)
            print(f"[cam_patch_probe_radar] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": comp_a, "sample B": comp_b}, {"mean abs diff": d})

        elif mode == "cam_patch_probe_feat":
            # introduce cam patches with feat only, no radar
            zero_radar(a)
            zero_radar(b)

            cam_mask = make_cam_mask(sample_positions(1))
            cam_input_a = denorm_img(a["cam_bev"], img_mean, img_std)
            cam_input_b = denorm_img(b["cam_bev"], img_mean, img_std)

            pred_a, mask = infer(a, cam_mask=cam_mask, use_radar=False, use_feat=True)
            pred_b, _    = infer(b, cam_mask=cam_mask, use_radar=False, use_feat=True)

            comp_a = make_composite(pred_a, cam_input_a, mask)
            comp_b = make_composite(pred_b, cam_input_b, mask)

            d = diff(comp_a, comp_b)
            print(f"[cam_patch_probe_feat] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": comp_a, "sample B": comp_b}, {"mean abs diff": d})

        # rad patch probes
        elif mode == "rad_patch_probe_radar":
            # introduce radar patches, no feat, predict radar
            rad_mask = make_rad_mask(sample_positions(10))
            ch = 1

            rad_input_a = a["radar_target"][:, ch:ch + 1]
            rad_input_b = b["radar_target"][:, ch:ch + 1]

            pred_a, mask, occ_a = infer_rad(a, rad_mask=rad_mask, use_feat=False, ch=ch)
            pred_b, _,    occ_b = infer_rad(b, rad_mask=rad_mask, use_feat=False, ch=ch)

            comp_a = make_rad_composite(pred_a, rad_input_a, mask, occ_a)
            comp_b = make_rad_composite(pred_b, rad_input_b, mask, occ_b)

            d = diff(comp_a, comp_b)
            print(f"[rad_patch_probe_radar] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": comp_a, "sample B": comp_b}, {"mean abs diff": d})

        elif mode == "rad_patch_probe_feat":
            # introduce radar patches with feat visible, predict radar
            rad_mask = make_rad_mask(sample_positions(10))
            ch = 1

            rad_input_a = a["radar_target"][:, ch:ch + 1]
            rad_input_b = b["radar_target"][:, ch:ch + 1]

            pred_a, mask, occ_a = infer_rad(a, rad_mask=rad_mask, use_feat=True, ch=ch)
            pred_b, _,    occ_b = infer_rad(b, rad_mask=rad_mask, use_feat=True, ch=ch)

            comp_a = make_rad_composite(pred_a, rad_input_a, mask, occ_a)
            comp_b = make_rad_composite(pred_b, rad_input_b, mask, occ_b)

            d = diff(comp_a, comp_b)
            print(f"[rad_patch_probe_feat] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": comp_a, "sample B": comp_b}, {"mean abs diff": d})

        # rad patch probe with target (incremental)
        elif mode in ("rad_patch_probe_with_target_radar", "rad_patch_probe_with_target_feat"):
            use_feat = mode == "rad_patch_probe_with_target_feat"

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

            steps = [0, 5, 10, 30, 50]
            N = grid_size[0] * grid_size[1]
            nw = grid_size[0]

            cam_input = denorm_img(a["cam_bev"], img_mean, img_std)

            for n in steps:
                cam_mask = torch.zeros(1, N, device=device)
                rad_mask = torch.ones(1, N, device=device)

                for row, col in positions[:n]:
                    rad_mask[:, row * nw + col] = 0

                task_masks = {
                    "cam_bev": cam_mask,
                    "radar": rad_mask,
                }
                if "bev_feat" in a:
                    task_masks["bev_feat"] = torch.zeros(1, N, device=device) if use_feat else torch.ones(1, N, device=device)

                preds, _ = model(
                    a,
                    mask_inputs=True,
                    task_masks=task_masks,
                    num_encoded_tokens=N + n,
                )

                pred = preds["radar"]
                target = a["radar_target"]
                mask_2d = rad_mask.reshape(1, grid_size[1], grid_size[0]).unsqueeze(1)

                mae = torch.abs(pred - target).mean().item()
                plot_radar_results(
                    mode,
                    pred,
                    target,
                    mask_2d,
                    {"mae": mae},
                    n_visible,
                    pred_occ_reg=False,
                )

                save_dir = f"reports/figures/diagnostic_plots/{mode}/visible_{n:03d}"
                img = cam_input.squeeze(0).detach().cpu().float().clamp(0, 1)
                if img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)

                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis("off")
                plt.savefig(os.path.join(save_dir, "camera_input.png"), dpi=200, bbox_inches="tight")
                plt.close()

        # feat patch probes
        elif mode == "feat_patch_probe_radar":
            # introduce feat patches with radar only, no cam, predict cam
            zero_radar(a)
            zero_radar(b)
            a["cam_bev"] = torch.zeros_like(a["cam_bev"])
            b["cam_bev"] = torch.zeros_like(b["cam_bev"])

            # restore radar from original samples
            a = to_device(collate_radar([ds[IDX_A]]))
            b = to_device(collate_radar([ds[IDX_B]]))

            feat_mask = make_feat_mask(sample_positions(10))
            cam_input_a = denorm_img(a["cam_bev"], img_mean, img_std)
            cam_input_b = denorm_img(b["cam_bev"], img_mean, img_std)

            pred_a, mask = infer_feat(a, feat_mask=feat_mask, use_radar=True, use_cam=False)
            pred_b, _    = infer_feat(b, feat_mask=feat_mask, use_radar=True, use_cam=False)

            d = diff(pred_a, pred_b)
            print(f"[feat_patch_probe_radar] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": pred_a, "sample B": pred_b}, {"mean abs diff": d})

        elif mode == "feat_patch_probe_cam":
            # introduce feat patches with cam fully visible, no radar, predict cam
            feat_mask = make_feat_mask(sample_positions(10))
            cam_input_a = denorm_img(a["cam_bev"], img_mean, img_std)
            cam_input_b = denorm_img(b["cam_bev"], img_mean, img_std)

            pred_a, mask = infer_feat(a, feat_mask=feat_mask, use_radar=False, use_cam=True)
            pred_b, _    = infer_feat(b, feat_mask=feat_mask, use_radar=False, use_cam=True)

            d = diff(pred_a, pred_b)
            print(f"[feat_patch_probe_cam] diff A vs B: {d:.6f}")
            plot_results(mode, {"sample A": pred_a, "sample B": pred_b}, {"mean abs diff": d})

        elif mode == "rad_from_camera":
            n_visible = 0

            rad_mask = make_rad_mask(sample_positions(n_visible))

            B = 1
            N = grid_size[0] * grid_size[1]

            task_masks = {
                "cam_bev": torch.zeros(B, N, device=device),  # camera visible
                "radar": rad_mask,                            # chosen radar patches visible
            }

            if "bev_feat" in a:
                task_masks["bev_feat"] = torch.ones(B, N, device=device)  # feat masked

            preds, _ = model(
                a,
                mask_inputs=True,
                task_masks=task_masks,
                num_encoded_tokens=N + n_visible,
            )

            pred = preds["radar"]
            target = a["radar_target"]
            mask_2d = rad_mask.reshape(1, grid_size[1], grid_size[0]).unsqueeze(1)

            mae = torch.abs(pred - target).mean().item()
            print(f"[rad_from_camera] visible radar patches: {n_visible} | mae: {mae:.6f}")

            plot_radar_results(
                mode,
                pred,
                target,
                mask_2d,
                {"mae": mae},
                n_visible,
                pred_occ_reg=False,
            )

            save_dir = f"reports/figures/diagnostic_plots/{mode}/visible_{n_visible:03d}"
            os.makedirs(save_dir, exist_ok=True)

            cam_input = denorm_img(a["cam_bev"], img_mean, img_std)
            img = cam_input.squeeze(0).detach().cpu().float().clamp(0, 1)

            if img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)

            img = img.numpy()

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img)

            pts = a["radar"]["points"].detach().cpu().numpy()
            pts = pts[pts[:, 0] == 0]

            x = pts[:, 1]
            y = pts[:, 2]

            point_cloud_range = [1.5, -6, -5, 28.5, 21, 20]
            x_min, y_min, _, x_max, y_max, _ = point_cloud_range

            H, W = img.shape[:2]

            px = (x - x_min) / (x_max - x_min) * (W - 1)
            py = (y - y_min) / (y_max - y_min) * (H - 1)

            inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)

            ax.scatter(px[inside], py[inside], s=2, c="lime", alpha=0.7)
            ax.axis("off")

            plt.savefig(
                os.path.join(save_dir, "camera_input.png"),
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close(fig)

        else:
            raise ValueError(f"Unknown mode: {mode}")