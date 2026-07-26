#!/usr/bin/env python3
"""
Visualize augmented samples and targets to verify augmentation correctness.
Creates a grid visualization of augmented data with target heatmaps.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bev_multimae.datasets.finetuning_data import BEVFineData


def visualize_sample_and_targets(
    sample_idx: int,
    dataset: BEVFineData,
    output_path: str = "augmentation_visualization.png",
    figsize: tuple = (16, 12),
):
    """
    Visualize augmented sample with targets and heatmaps.
    
    Args:
        sample_idx: Index of sample to visualize
        dataset: FinetuningDataset instance
        output_path: Path to save visualization PNG
        figsize: Figure size (width, height)
    """
    print(f"\nLoading sample {sample_idx}...")
    sample = dataset[sample_idx]
    
    cam_bev = sample["cam_bev"]  # [C, H, W]
    boxes = sample["boxes"]  # List of [N, 7]
    radar = sample["radar"]
    targets = sample.get("targets", {})
    
    print(f"  cam_bev shape: {cam_bev.shape}")
    print(f"  boxes: {len(boxes)} boxes")
    print(f"  radar points: {radar['points'].shape if isinstance(radar, dict) else 'N/A'}")
    if targets:
        print(f"  target keys: {list(targets.keys())}")
        if 'heatmap' in targets:
            print(f"    heatmap shape: {targets['heatmap'].shape}")
    
    # Prepare visualization
    num_rows = 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    idx = 0
    
    # 1. Camera BEV image
    ax = axes[idx]
    if cam_bev.shape[0] == 3:
        img = cam_bev.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # Normalize to [0, 1]
        # Flip vertically to match radar/target coordinate system
        img = np.flip(img, axis=0)
        ax.imshow(img)
    else:
        img = cam_bev[0].numpy()
        img = np.flip(img, axis=0)
        ax.imshow(img, cmap='gray')
    ax.set_title("Camera BEV Image (V-flipped)", fontsize=12, fontweight='bold')
    ax.axis('off')
    idx += 1
    
    # 2. Target heatmap
    if targets and 'heatmap' in targets:
        ax = axes[idx]
        heatmap = targets['heatmap']  # [C, H, W], usually C=1 for single class
        if heatmap.dim() == 3 and heatmap.shape[0] > 0:
            hm = heatmap[0].numpy()  # Take first channel
        else:
            hm = heatmap.numpy() if heatmap.dim() == 2 else heatmap[0].numpy()
        im = ax.imshow(hm, cmap='hot', origin='lower')
        plt.colorbar(im, ax=ax, label='Confidence')
        ax.set_title("Target Heatmap", fontsize=12, fontweight='bold')
        ax.set_xlabel("X [grid cells]")
        ax.set_ylabel("Y [grid cells]")
    idx += 1
    
    # 3. Target regression (offset)
    if targets and 'reg' in targets:
        ax = axes[idx]
        reg = targets['reg']  # [C, H, W]
        if reg.dim() == 3 and reg.shape[0] >= 2:
            reg_mag = torch.sqrt(reg[0]**2 + reg[1]**2).numpy()
        else:
            reg_mag = reg[0].numpy() if reg.dim() == 3 else reg.numpy()
        im = ax.imshow(reg_mag, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, label='Offset Magnitude')
        ax.set_title("Target Regression (Offset)", fontsize=12, fontweight='bold')
        ax.set_xlabel("X [grid cells]")
        ax.set_ylabel("Y [grid cells]")
    idx += 1
    
    # 4. Target height
    if targets and 'height' in targets:
        ax = axes[idx]
        height = targets['height']  # [C, H, W]
        h = height[0].numpy() if height.dim() == 3 else height.numpy()
        im = ax.imshow(h, cmap='plasma', origin='lower')
        plt.colorbar(im, ax=ax, label='Height')
        ax.set_title("Target Height", fontsize=12, fontweight='bold')
        ax.set_xlabel("X [grid cells]")
        ax.set_ylabel("Y [grid cells]")
    idx += 1
    
    # 5. Target dimensions
    if targets and 'dim' in targets:
        ax = axes[idx]
        dim = targets['dim']  # [C, H, W]
        if dim.dim() == 3 and dim.shape[0] >= 3:
            dim_mag = torch.sqrt(dim[0]**2 + dim[1]**2 + dim[2]**2).numpy()
        else:
            dim_mag = dim[0].numpy() if dim.dim() == 3 else dim.numpy()
        im = ax.imshow(dim_mag, cmap='cool', origin='lower')
        plt.colorbar(im, ax=ax, label='Dimension Magnitude')
        ax.set_title("Target Dimensions", fontsize=12, fontweight='bold')
        ax.set_xlabel("X [grid cells]")
        ax.set_ylabel("Y [grid cells]")
    idx += 1
    
    # 6. Target rotation
    if targets and 'rot' in targets:
        ax = axes[idx]
        rot = targets['rot']  # [C, H, W]
        r = rot[0].numpy() if rot.dim() == 3 else rot.numpy()
        im = ax.imshow(r, cmap='hsv', origin='lower')
        plt.colorbar(im, ax=ax, label='Rotation (rad)')
        ax.set_title("Target Rotation", fontsize=12, fontweight='bold')
        ax.set_xlabel("X [grid cells]")
        ax.set_ylabel("Y [grid cells]")
    idx += 1
    
    # 7. Target mask
    if targets and 'masks' in targets:
        ax = axes[idx]
        masks = targets['masks']  # [C, H, W]
        m = masks[0].numpy() if masks.dim() == 3 else masks.numpy()
        ax.imshow(m, cmap='binary', origin='lower')
        ax.set_title("Target Mask (Valid Cells)", fontsize=12, fontweight='bold')
        ax.set_xlabel("X [grid cells]")
        ax.set_ylabel("Y [grid cells]")
    idx += 1
    
    # 8. Radar points projection
    if radar and 'points' in radar:
        ax = axes[idx]
        points = radar['points']
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        
        # Project radar points to BEV (64x64 grid)
        if hasattr(dataset, 'pillarizer') and hasattr(dataset.pillarizer, 'point_cloud_range'):
            pcr = dataset.pillarizer.point_cloud_range
            if isinstance(pcr, torch.Tensor):
                pcr = pcr.tolist()
        else:
            # Default from meta
            pcr = dataset.meta.get('pcr', [-51.2, -51.2, -5, 51.2, 51.2, 3])
            if isinstance(pcr, torch.Tensor):
                pcr = pcr.tolist()
        
        # Grid spans [pcr[0]:pcr[3]] in x and [pcr[1]:pcr[4]] in y
        x_min, y_min, z_min = pcr[0], pcr[1], pcr[2]
        x_max, y_max, z_max = pcr[3], pcr[4], pcr[5]
        
        grid_size = 64
        x_norm = (points[:, 1] - x_min) / (x_max - x_min)  # Normalize x to [0, 1]
        y_norm = (points[:, 2] - y_min) / (y_max - y_min)  # Normalize y to [0, 1]
        
        # Clip to valid range
        x_norm = np.clip(x_norm, 0, 1)
        y_norm = np.clip(y_norm, 0, 1)
        
        # Create radar point plot
        x_grid = x_norm * grid_size
        y_grid = y_norm * grid_size

        ax.scatter(x_grid, y_grid, s=12, c='lime', alpha=0.8)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_title("Radar Points Projection (64x64)", fontsize=12, fontweight='bold')
        ax.set_xlabel("X [grid cells]")
        ax.set_ylabel("Y [grid cells]")
    idx += 1
    
    # 9. Sample info text
    ax = axes[idx]
    ax.axis('off')
    
    # Get point cloud range
    if hasattr(dataset, 'pillarizer') and hasattr(dataset.pillarizer, 'point_cloud_range'):
        pcr = dataset.pillarizer.point_cloud_range
        if isinstance(pcr, torch.Tensor):
            pcr = pcr.tolist()
    else:
        pcr = dataset.meta.get('pcr', [-51.2, -51.2, -5, 51.2, 51.2, 3])
        if isinstance(pcr, torch.Tensor):
            pcr = pcr.tolist()
    
    x_center = (pcr[0] + pcr[3]) / 2
    y_center = (pcr[1] + pcr[4]) / 2
    
    info_text = f"""
Sample Index: {sample_idx}
Augmentation Enabled: {dataset.augment}

Point Cloud Range:
  X: [{pcr[0]:.1f}, {pcr[3]:.1f}]
  Y: [{pcr[1]:.1f}, {pcr[4]:.1f}]
  Z: [{pcr[2]:.1f}, {pcr[5]:.1f}]

Grid Center:
  X: {x_center:.1f}
  Y: {y_center:.1f}

Augmentation Settings:
  H-flip rate: {dataset.h_flip_rate if hasattr(dataset, 'h_flip_rate') else 'N/A'}
  V-flip rate: {dataset.v_flip_rate if hasattr(dataset, 'v_flip_rate') else 'N/A'}
  Rotation rate: {dataset.rot_rate if hasattr(dataset, 'rot_rate') else 'N/A'}
  Rotation angle: {dataset.rot_angle if hasattr(dataset, 'rot_angle') else 'N/A'}

Data Shapes:
  Cam BEV: {tuple(cam_bev.shape)}
  Boxes: {len(boxes)} boxes
  Radar points: {radar['points'].shape if isinstance(radar, dict) and 'points' in radar else 'N/A'}
  Heatmap: {targets.get('heatmap', torch.tensor([])).shape if targets else 'N/A'}
"""
    ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', 
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    plt.close()


def main():
    """Load dataset and visualize multiple samples."""
    project_root = Path(__file__).parent
    
    # Load meta to get point_cloud_range
    pretrain_path = str(project_root / "data" / "processed_2" / "right")
    meta = torch.load(os.path.join(pretrain_path, "meta.pt"), map_location="cpu", weights_only=False)
    point_cloud_range = meta.get("pcr", None)  # Use 'pcr' key instead of 'point_cloud_range'
    
    print(f"Point cloud range from meta: {point_cloud_range}")
    
    # Create dataset with augmentation enabled
    print("Creating BEVFineData with augmentation enabled...")
    dataset = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=str(project_root / "data_finetuning"),
        direction="right",
        split="train",
        augment=True,  # Enable augmentation
        img_mean=None,
        img_std=None,
        point_cloud_range=point_cloud_range,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty! Check data paths.")
        print(f"  pretrain_path: {project_root / 'data' / 'processed_2' / 'right'}")
        print(f"  finetune_path: {project_root / 'data_finetuning'}")
        return
    
    # Visualize first few samples
    num_samples = min(3, len(dataset))
    output_dir = project_root
    
    for i in range(num_samples):
        output_path = str(output_dir / f"augmentation_sample_{i}.png")
        try:
            visualize_sample_and_targets(i, dataset, output_path=output_path)
        except Exception as e:
            print(f"ERROR visualizing sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Visualization complete! Saved {num_samples} PNG files to {output_dir}")


if __name__ == "__main__":
    main()
