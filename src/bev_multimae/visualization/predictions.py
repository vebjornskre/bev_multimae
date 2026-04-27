import matplotlib.pyplot as plt
import numpy as np
import os

def dbsm_to_m2(rcs_dbsm):
    return 10**(rcs_dbsm / 10)

def minmax_per_channel(img):
    out = np.zeros_like(img)
    for c in range(img.shape[-1]):
        mn, mx = img[..., c].min(), img[..., c].max()
        out[..., c] = (img[..., c] - mn) / (mx - mn + 1e-6)
    return out

def viz_preds(preds, batch, folder, radar_channel=None):
    save_folder = os.path.join(folder, 'predictions')
    os.makedirs(save_folder, exist_ok=True)

    for k, v in preds.items():

        if k == "radar":
            
            pred = v[0].detach().cpu().permute(1, 2, 0).numpy()
            inp  = batch["radar_target"][0].cpu().permute(1, 2, 0).numpy()

            # Sigmoid to collaps to occupied or not for the first channel
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
            ]

            for title, ch in zip(rad_titles, channels):
                pred_ch = pred[..., ch]
                inp_ch  = inp[..., ch]

                if ch == 0:
                    plot_pred = (pred_ch > 0.5).astype(np.float32)
                    v_min, v_max = 0, 1
                else:
                    occ_mask = inp[..., 0] > 0.5
                    inp_ch  = np.where(occ_mask, inp_ch, np.nan)
                    pred_ch = np.where(occ_mask, pred_ch, np.nan)
                    plot_pred = pred_ch
                    v_min, v_max = np.nanmin(inp_ch), np.nanmax(inp_ch)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                axes[0].imshow(inp_ch, cmap='gray', origin='lower', vmin=v_min, vmax=v_max)
                
                axes[0].set_title(f"Input: {title}")
                axes[0].set_xlabel("Forward (m)")
                axes[0].set_ylabel("Left (m)")

                axes[1].imshow(plot_pred, cmap='gray', origin='lower', vmin=v_min, vmax=v_max)
                axes[1].set_title(f"Prediction: {title}")

                for ax in axes:
                    ax.axis('off')

                name = f"radar_ch{ch}.png" if radar_channel is None else f"radar_ch{radar_channel}.png"
                plt.savefig(os.path.join(save_folder, name), bbox_inches='tight', pad_inches=0)
                plt.close()

        else:  # cam_bev
            pred = v[0].detach().cpu().permute(1, 2, 0).numpy()
            inp  = batch["cam_bev"][0].cpu().permute(1, 2, 0).numpy()

            pred = minmax_per_channel(pred)
            inp  = minmax_per_channel(inp)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(inp[..., :3], origin="lower")
            axes[0].set_title("Input")
            axes[1].imshow(pred[..., :3], origin="lower")
            axes[1].set_title("Prediction")

        for ax in axes:
            ax.axis('off')

        plt.savefig(os.path.join(save_folder, f'{k}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
