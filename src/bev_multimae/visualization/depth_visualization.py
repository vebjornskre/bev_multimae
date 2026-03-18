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

def plot_depth_maps(cfg, img, depth, feat=None):

    if feat is not None:
        # We dowsample to the size of feat
        target_size = feat.shape[-2:]

        # Bilinear downsampling to feature maps resolution
        depth_ds = F.interpolate(depth, size=target_size, mode='bilinear', align_corners=False, antialias=True)

        # Full image and depth map
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))

        W, H = img.size

        # Image
        if type(img) == torch.Tensor:
            img.cpu().numpy()
        axes[0].imshow(img)
        axes[0].set_title(f"Original Image (3x{H}x{W})")
        axes[0].axis("off")
        
        # Depth
        if type(depth) == torch.Tensor:
            depth_np = depth.squeeze().cpu().numpy()
        else:
            depth_np = depth
        axes[1].imshow(depth_np, cmap='plasma')
        axes[1].set_title("Full Resolution Depth")
        axes[1].axis("off")
        
        plt.savefig(f"{save_dir}/vis_full.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Feature map and downsampled depth
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        
        feat = feat.squeeze(0)

        C, W, H = feat.size()

        # Average all channels or pick a channel
        feat_img = feat.mean(dim=0).cpu()
        # feat_img = feat[0,...]

        axes[0].imshow(feat_img, cmap='viridis')
        axes[0].set_title(f"Feature Map (Mean({C})x{W}x{H})")
        axes[0].axis("off")

        # Downsampled depth
        depth_ds_np = depth_ds.squeeze().cpu().numpy()
        axes[1].imshow(depth_ds_np, cmap='plasma')
        axes[1].set_title("Downsampled Depth")
        axes[1].axis("off")

        plt.savefig(f"{save_dir}/vis_downsampled.png", dpi=200, bbox_inches="tight")
        plt.close()

        print(f'Figures saved: {save_dir}/vis_full.png, {save_dir}/vis_downsampled.png')

    else:
        save_dir = os.path.join(Path(to_absolute_path(cfg.plot_folder)), 'depth_imgs')
        fname = f'depth_map_{cfg.depth_model}.png'
        save_path = os.path.join(save_dir, fname)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Depth map
        if type(depth) == torch.Tensor:
            depth_np = depth.squeeze().cpu().numpy()
        else:
            depth_np = depth
        im = axes[1].imshow(depth_np, cmap='plasma')
        plt.colorbar(im, ax=axes[1], label='Depth (m)')
        axes[1].set_title("Depth Map")
        axes[1].axis("off")

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f'Figure saved: {save_path}')


@hydra.main(config_path="../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:

    # CODE BELOW DEPRECIATED SWITCH TO NEW METHOD USING THE DEPTH ESTIMATOR CLASS

    # device = 'cpu'
    # zoe, _, img = load_zoe(cfg, zoe=True, cnn=False, device=device)

    # Path from cfg
    # plot_folder = os.path.join(Path(to_absolute_path(cfg.plot_folder)), 'depth_imgs')

    # Run depth estimation
    # depth = zoe_depth(model=zoe, img=img)
    # plot_depth_maps(plot_folder, img, depth)
    ...
if __name__ == '__main__':
    main()