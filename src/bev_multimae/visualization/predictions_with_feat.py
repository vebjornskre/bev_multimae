import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA


def minmax_per_channel(img):
    out = np.zeros_like(img)
    for c in range(img.shape[-1]):
        mn, mx = img[..., c].min(), img[..., c].max()
        out[..., c] = (img[..., c] - mn) / (mx - mn + 1e-6)
    return out


def pca_pair(a, b, mask_empty=True, boost=1.0):
    if a.ndim == 3:
        a = np.transpose(a, (1, 2, 0))
    if b.ndim == 3:
        b = np.transpose(b, (1, 2, 0))

    H, W, C = a.shape
    af = a.reshape(-1, C).astype(np.float32)
    bf = b.reshape(-1, C).astype(np.float32)

    va = np.isfinite(af).all(axis=1)
    vb = np.isfinite(bf).all(axis=1)

    if mask_empty:
        va &= np.linalg.norm(af, axis=1) > 1e-6
        vb &= np.linalg.norm(bf, axis=1) > 1e-6

    fit_mask = va | vb
    x_fit = np.concatenate([af[fit_mask], bf[fit_mask]], axis=0)

    ar = np.zeros((af.shape[0], 3), dtype=np.float32)
    br = np.zeros((bf.shape[0], 3), dtype=np.float32)

    if x_fit.shape[0] < 4:
        return ar.reshape(H, W, 3), br.reshape(H, W, 3)

    pca = PCA(n_components=3).fit(x_fit)

    ap = pca.transform(af[va])
    bp = pca.transform(bf[vb])

    both = np.concatenate([ap, bp], axis=0)
    lo = np.percentile(both, 1, axis=0)
    hi = np.percentile(both, 99, axis=0)

    ap = np.clip((ap - lo) / (hi - lo + 1e-8), 0, 1)
    bp = np.clip((bp - lo) / (hi - lo + 1e-8), 0, 1)

    ap = np.clip((ap - 0.5) * boost + 0.5, 0, 1)
    bp = np.clip((bp - 0.5) * boost + 0.5, 0, 1)

    ar[va] = ap
    br[vb] = bp

    return ar.reshape(H, W, 3), br.reshape(H, W, 3)

def scale_pair(a, b, q=(1, 99)):
    x = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    vmin, vmax = np.percentile(x, q)
    a = np.clip((a - vmin) / (vmax - vmin + 1e-8), 0, 1)
    b = np.clip((b - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return a, b, vmin, vmax


def viz_preds(preds, batch, folder, radar_channel=None, point_cloud_range=None):
    save_folder = os.path.join(folder, "predictions")
    os.makedirs(save_folder, exist_ok=True)

    for k, v in preds.items():

        if k == "radar":
            pred = v[0].detach().cpu().permute(1, 2, 0).numpy()
            inp = batch["radar_target"][0].detach().cpu().permute(1, 2, 0).numpy()

            pred[..., 0] = 1 / (1 + np.exp(-pred[..., 0]))

            C = pred.shape[-1]
            channels = [radar_channel] if radar_channel is not None else range(C)

            rad_titles = [
                "Occupancy",
                "Density (log count)",
                "Height (mean z)",
                "Velocity (mean)",
                "RCS (mean)",
                "Height (var)",
                "Velocity (var)",
                "RCS (var)",
                "SNR (mean)",
                "Mean x from pillar center",
                "Mean y from pillar center",
            ]
            rad_units = [
                None,
                "log count",
                "m",
                "m/s",
                "dBsm",
                "m$^2$",
                "(m/s)$^2$",
                "dBsm$^2$",
                "dB",
                "m",
                "m",
            ]

            fig, axes = plt.subplots(3, 4, figsize=(16, 10))
            axes = axes.flatten()
            fig.suptitle("Radar target and camera BEV", fontsize=24)

            cam_bev = batch["cam_bev"][0].detach().cpu().permute(1, 2, 0).numpy()
            cam_bev = np.clip(cam_bev[..., :3], 0, 1)

            axes[0].imshow(cam_bev, origin="lower")

            if point_cloud_range is not None:
                pts = batch["radar"]["points"].detach().cpu().numpy()
                pts = pts[pts[:, 0] == 0]

                x = pts[:, 1]
                y = pts[:, 2]

                x_min, y_min, _, x_max, y_max, _ = point_cloud_range
                H, W = cam_bev.shape[:2]

                px = (x - x_min) / (x_max - x_min) * (W - 1)
                py = (y - y_min) / (y_max - y_min) * (H - 1)

                inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                axes[0].scatter(px[inside], py[inside], s=2, c="lime", alpha=0.7)

            axes[0].set_title("Camera BEV with radar points")
            axes[0].axis("off")

            for ch in range(C):
                title = rad_titles[ch] if ch < len(rad_titles) else f"Channel {ch}"
                unit = rad_units[ch] if ch < len(rad_units) else None

                inp_ch = inp[..., ch]

                if ch == 0:
                    plot_inp = (inp_ch > 0.5).astype(np.float32)
                    v_min, v_max = 0, 1
                else:
                    occ_mask = inp[..., 0] > 0.5
                    plot_inp = np.where(occ_mask, inp_ch, np.nan)
                    v_min = np.nanmin(plot_inp)
                    v_max = np.nanmax(plot_inp)

                ax = axes[ch +1]
                im = ax.imshow(plot_inp, cmap="jet", origin="lower", vmin=v_min, vmax=v_max)
                ax.set_title(title)
                ax.axis("off")

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if ch == 0:
                    cbar.set_ticks([0, 1])
                    cbar.set_ticklabels(["0", "1"])
                elif unit is not None:
                    cbar.set_label(unit)


            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(save_folder, "radar_targets.png"), bbox_inches="tight", pad_inches=0.08)
            plt.close()

            for ch in channels:
                title = rad_titles[ch] if ch < len(rad_titles) else f"Channel {ch}"

                pred_ch = pred[..., ch]
                inp_ch = inp[..., ch]

                if ch == 0:
                    inp_ch = (inp_ch > 0.5).astype(np.float32)
                    plot_pred = (pred_ch > 0.5).astype(np.float32)
                    v_min, v_max = 0, 1
                else:
                    occ_mask = inp[..., 0] > 0.5
                    inp_ch = np.where(occ_mask, inp_ch, np.nan)
                    pred_ch = np.where(occ_mask, pred_ch, np.nan)
                    plot_pred = pred_ch
                    v_min = np.nanmin(inp_ch)
                    v_max = np.nanmax(inp_ch)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                im0 = axes[0].imshow(inp_ch, cmap="jet", origin="lower", vmin=v_min, vmax=v_max)
                axes[0].set_title(f"Input: {title}")

                axes[1].imshow(plot_pred, cmap="jet", origin="lower", vmin=v_min, vmax=v_max)
                axes[1].set_title(f"Prediction: {title}")

                for ax in axes:
                    ax.axis("off")

                fig.subplots_adjust(right=0.88)
                cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])

                cbar = fig.colorbar(im0, cax=cbar_ax)
                if ch == 0:
                    cbar.set_ticks([0, 1])
                    cbar.set_ticklabels(["0", "1"])

                name = f"radar_ch{ch}.png"
                plt.savefig(os.path.join(save_folder, name), bbox_inches="tight", pad_inches=0.08)
                plt.close()

        elif k == "bev_feat":
            pred = v[0].detach().cpu().float().numpy()
            inp = batch["bev_feat"][0].detach().cpu().float().numpy()

            inp_rgb, pred_rgb = pca_pair(inp, pred, mask_empty=True, boost=1.0)

            pred_energy = np.log1p(np.abs(pred).sum(axis=0))
            inp_energy = np.log1p(np.abs(inp).sum(axis=0))

            inp_energy, pred_energy, e_min, e_max = scale_pair(inp_energy, pred_energy)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(inp_rgb, origin="lower")
            axes[0, 0].set_title("Input BEV feature PCA")

            axes[0, 1].imshow(pred_rgb, origin="lower")
            axes[0, 1].set_title("Prediction BEV feature PCA")

            im0 = axes[1, 0].imshow(inp_energy, origin="lower", cmap="magma", vmin=0, vmax=1)
            axes[1, 0].set_title(f"Input energy shared [{e_min:.3g}, {e_max:.3g}]")

            im1 = axes[1, 1].imshow(pred_energy, origin="lower", cmap="magma", vmin=0, vmax=1)
            axes[1, 1].set_title("Prediction energy shared")

            for ax in axes.flat:
                ax.axis("off")

            fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, "bev_feat.png"), bbox_inches="tight", pad_inches=0.08)
            plt.close()

        else:
            pred = v[0].detach().cpu().permute(1, 2, 0).numpy()
            inp = batch[k][0].detach().cpu().permute(1, 2, 0).numpy()

            pred = np.clip(pred, 0, 1)
            inp = np.clip(inp, 0, 1)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(inp[..., :3], origin="lower")
            axes[0].set_title(f"Input: {k}")

            axes[1].imshow(pred[..., :3], origin="lower")
            axes[1].set_title(f"Prediction: {k}")

            for ax in axes:
                ax.axis("off")

            plt.savefig(os.path.join(save_folder, f"{k}.png"), bbox_inches="tight", pad_inches=0.08)
            plt.close()