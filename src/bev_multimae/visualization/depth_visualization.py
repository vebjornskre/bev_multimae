import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import to_absolute_path
import hydra
import torch.nn.functional as F
from PIL import Image
import os
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

MODEL_NAMES = {
    "metric3d": "Metric3D",
    "depth_any": "Depth Anything",
    "moge": "MoGe-2",
    "depth_any_rel": "Depth Anything Relative",
    "depth_pro": "Depth Pro",
    "unidepth": "UniDepth",
    "zoe" : "ZoeDepth"
}

def plot_depth_maps(cfg, img, depth, feat=None):

    if feat is not None:
        target_size = feat.shape[-2:]
        depth_ds = F.interpolate(depth, size=target_size, mode='bilinear', align_corners=False, antialias=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        W, H = img.size

        if type(img) == torch.Tensor:
            img.cpu().numpy()
        axes[0].imshow(img)
        axes[0].set_title(f"Original Image (3x{H}x{W})")
        axes[0].axis("off")
        
        if type(depth) == torch.Tensor:
            depth_np = depth.squeeze().cpu().numpy()
        else:
            depth_np = depth
        axes[1].imshow(depth_np, cmap='plasma')
        axes[1].set_title("Full Resolution Depth")
        axes[1].axis("off")
        
        plt.savefig(f"{save_dir}/vis_full.png", dpi=200, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        feat = feat.squeeze(0)
        C, W, H = feat.size()
        feat_img = feat.mean(dim=0).cpu()

        axes[0].imshow(feat_img, cmap='viridis')
        axes[0].set_title(f"Feature Map (Mean({C})x{W}x{H})")
        axes[0].axis("off")

        depth_ds_np = depth_ds.squeeze().cpu().numpy()
        axes[1].imshow(depth_ds_np, cmap='plasma')
        axes[1].set_title("Downsampled Depth")
        axes[1].axis("off")

        plt.savefig(f"{save_dir}/vis_downsampled.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f'Figures saved: {save_dir}/vis_full.png, {save_dir}/vis_downsampled.png')

    else:
        save_dir = os.path.join(Path(to_absolute_path(cfg.plot_folder)), 'depth_imgs')
        os.makedirs(save_dir, exist_ok=True)

        model_name = MODEL_NAMES.get(cfg.depth_model, str(cfg.depth_model))

        fname = f'depth_map_{model_name}.png'
        save_path = os.path.join(save_dir, fname)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].imshow(img)
        axes[0].set_title("Original Image")

        if type(depth) == torch.Tensor:
            depth_np = depth.squeeze().cpu().numpy()
        else:
            depth_np = depth
        im = axes[1].imshow(depth_np, cmap='plasma')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Depth (m)')
        axes[1].set_title("Depth Map")
        axes[1].set_xlabel("u (px)", fontsize=14)
        axes[1].set_ylabel("v (px)", fontsize=14)
        axes[1].tick_params(axis="both", labelsize=12)

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f'Figure saved: {save_path}')

        # Save raw depth
        npy_path = save_path.replace('.png', '.npy')
        np.save(npy_path, depth_np)
        print(f'Depth saved: {npy_path}')

        # Save depth as individual PNG with colorbar and title
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        im = ax.imshow(depth_np, cmap='plasma')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)

        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Depth (m)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        ax.set_title(model_name, fontsize=15) #, fontweight='bold')
        ax.set_xlabel("u (px)", fontsize=14)
        ax.set_ylabel("v (px)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

        individual_path = save_path.replace('.png', '_individual.png')
        plt.savefig(individual_path, dpi=200)
        plt.close()
        print(f'Individual depth saved: {individual_path}')

        # Save RGB once as individual PNG with title
        rgb_path = os.path.join(save_dir, 'rgb.png')
        if not os.path.exists(rgb_path):
            img.save(rgb_path)
            print(f'RGB saved: {rgb_path}')

        rgb_individual_path = os.path.join(save_dir, 'rgb_individual.png')
        if not os.path.exists(rgb_individual_path):
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.imshow(img)
            ax.set_title("Original", fontsize=14, fontweight='bold')
            plt.savefig(rgb_individual_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f'RGB individual saved: {rgb_individual_path}')


@hydra.main(config_path="../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    ...

if __name__ == '__main__':
    main()