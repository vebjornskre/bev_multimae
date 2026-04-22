import matplotlib.pyplot as plt
import numpy as np
import os

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
            axes[0].imshow(inp[..., 0], cmap='gray', origin='lower')
            axes[0].set_title("Input (occupancy)")
            axes[1].imshow(pred[..., 0], cmap='gray', origin='lower')
            axes[1].set_title("Prediction (occupancy)")

        else:  # cam_bev
            pred = v[0].detach().cpu().permute(1, 2, 0).numpy()
            inp = batch["cam_bev"][0].cpu().permute(1, 2, 0).numpy()

            print(pred.mean(), pred.std())

            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
            inp  = (inp  - inp.min())  / (inp.max()  - inp.min()  + 1e-6)

            print(pred.mean(), pred.std())

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(inp[..., :3], origin="lower")
            axes[0].set_title("Input")
            axes[1].imshow(pred[..., :3], origin="lower")
            axes[1].set_title("Prediction")

        for ax in axes:
            ax.axis('off')

        plt.savefig(os.path.join(save_folder, f'{k}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
