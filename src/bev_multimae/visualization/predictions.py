import matplotlib.pyplot as plt
import numpy as np
import os

def minmax_per_channel(img):
    out = np.zeros_like(img)
    for c in range(img.shape[-1]):
        mn, mx = img[..., c].min(), img[..., c].max()
        out[..., c] = (img[..., c] - mn) / (mx - mn + 1e-6)
    return out

def viz_preds(preds, batch, folder):
    save_folder = os.path.join(folder, 'predictions')
    os.makedirs(save_folder, exist_ok=True)

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    for k, v in preds.items():

        if k == "radar":
            pred = v[0].detach().cpu().permute(1, 2, 0).numpy()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)

            inp = batch["radar_target"][0].cpu().permute(1, 2, 0).numpy()
            inp = (inp - inp.min()) / (inp.max() - inp.min() + 1e-6)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].set_xlabel("Forward (m)")
            axes[0].set_ylabel("Left (m)")
            axes[0].imshow(inp[..., 1], cmap='gray', origin='lower')
            axes[0].set_title("Input (log density)")
            axes[1].imshow(pred[..., 1], cmap='gray', origin='lower')
            axes[1].set_title("Prediction (log density)")

        else:  # cam_bev
            pred = v[0].detach().cpu().permute(1, 2, 0).numpy()
            inp  = batch["cam_bev"][0].cpu().permute(1, 2, 0).numpy()

            print(f"Raw pred - mean: {pred.mean():.4f}, std: {pred.std():.4f}, min: {pred.min():.4f}, max: {pred.max():.4f}")
            print(f"Raw inp  - mean: {inp.mean():.4f}, std: {inp.std():.4f}, min: {inp.min():.4f}, max: {inp.max():.4f}")

            pred_denorm = pred * imagenet_std + imagenet_mean
            inp_denorm  = inp  * imagenet_std + imagenet_mean
            
            print(f"After denorm pred - mean: {pred_denorm.mean():.4f}, std: {pred_denorm.std():.4f}, min: {pred_denorm.min():.4f}, max: {pred_denorm.max():.4f}")
            print(f"After denorm inp  - mean: {inp_denorm.mean():.4f}, std: {inp_denorm.std():.4f}, min: {inp_denorm.min():.4f}, max: {inp_denorm.max():.4f}")

            pred = minmax_per_channel(pred_denorm)
            inp  = minmax_per_channel(inp_denorm)

            print(f"After minmax pred - mean: {pred.mean():.4f}, std: {pred.std():.4f}, min: {pred.min():.4f}, max: {pred.max():.4f}")

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(inp[..., :3], origin="lower")
            axes[0].set_title("Input")
            axes[1].imshow(pred[..., :3], origin="lower")
            axes[1].set_title("Prediction")

        for ax in axes:
            ax.axis('off')

        plt.savefig(os.path.join(save_folder, f'{k}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
