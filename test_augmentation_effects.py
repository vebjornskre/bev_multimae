#!/usr/bin/env python3
"""
Test augmentation effects by comparing augmented vs non-augmented versions.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bev_multimae.datasets.finetuning_data import BEVFineData


def plot_augmentation_comparison():
    """Compare augmented vs non-augmented versions of the same sample."""
    project_root = Path(__file__).parent
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load meta
    pretrain_path = str(project_root / "data" / "processed_2" / "right")
    meta = torch.load(os.path.join(pretrain_path, "meta.pt"), map_location="cpu", weights_only=False)
    point_cloud_range = meta.get("pcr", None)
    
    finetune_path = str(project_root / "data_finetuning")
    
    # Create dataset WITHOUT augmentation
    print("Creating dataset WITHOUT augmentation...")
    dataset_no_aug = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=finetune_path,
        direction="right",
        split="train",
        augment=False,
        img_mean=None,
        img_std=None,
        point_cloud_range=point_cloud_range,
    )
    
    # Create dataset WITH augmentation
    print("Creating dataset WITH augmentation...")
    dataset_with_aug = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=finetune_path,
        direction="right",
        split="train",
        augment=True,
        h_flip_rate=1.0,  # Force H-flip
        v_flip_rate=0.0,
        rot_rate=0.0,
        img_mean=None,
        img_std=None,
        point_cloud_range=point_cloud_range,
    )
    
    # Load the same sample from both datasets
    sample_idx = 0
    print(f"\nLoading sample {sample_idx} from both datasets...")
    
    sample_no_aug = dataset_no_aug[sample_idx]
    sample_with_aug = dataset_with_aug[sample_idx]
    
    # Extract data
    cam_no_aug = sample_no_aug["cam_bev"]
    cam_aug = sample_with_aug["cam_bev"]
    
    heatmap_no_aug = sample_no_aug["targets"]["heatmap"][0].numpy()
    heatmap_aug = sample_with_aug["targets"]["heatmap"][0].numpy()
    
    boxes_no_aug = sample_no_aug["boxes"]
    boxes_aug = sample_with_aug["boxes"]
    
    print(f"  Boxes (no aug): {[b.shape for b in boxes_no_aug if b is not None]}")
    print(f"  Boxes (with aug): {[b.shape for b in boxes_aug if b is not None]}")
    
    # Print box coordinates to verify flipping
    if boxes_no_aug and boxes_no_aug[0] is not None:
        print(f"  Original box X: {boxes_no_aug[0][0, 0]:.4f}, Y: {boxes_no_aug[0][0, 1]:.4f}, Angle: {boxes_no_aug[0][0, 6]:.4f}")
    if boxes_aug and boxes_aug[0] is not None:
        print(f"  Augmented box X: {boxes_aug[0][0, 0]:.4f}, Y: {boxes_aug[0][0, 1]:.4f}, Angle: {boxes_aug[0][0, 6]:.4f}")
        
        # For H-flip, Y should be: 2*y_center - y_original
        # y_center = (pcr[1] + pcr[4]) / 2 = (-6 + 21) / 2 = 7.5
        y_center = (point_cloud_range[1] + point_cloud_range[4]) / 2
        y_expected = 2 * y_center - boxes_no_aug[0][0, 1]
        y_actual = boxes_aug[0][0, 1]
        print(f"  Expected Y after H-flip: {y_expected:.4f}")
        print(f"  Actual Y after H-flip:   {y_actual:.4f}")
        print(f"  Difference: {abs(y_expected - y_actual):.6f} (should be ~0)")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: No augmentation
    # Camera
    ax = axes[0, 0]
    cam_img = cam_no_aug.permute(1, 2, 0).numpy()
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    cam_img = np.flip(cam_img, axis=0)
    ax.imshow(cam_img)
    ax.set_title("Camera (No Aug)", fontweight='bold')
    ax.axis('off')
    
    # Heatmap
    ax = axes[0, 1]
    ax.imshow(heatmap_no_aug, cmap='hot', origin='lower')
    ax.set_title("Heatmap (No Aug)", fontweight='bold')
    ax.set_xlabel("X [grid]")
    ax.set_ylabel("Y [grid]")
    
    # Box info
    ax = axes[0, 2]
    ax.axis('off')
    if boxes_no_aug and boxes_no_aug[0] is not None:
        box_info = f"""
Original (No Aug):
X: {boxes_no_aug[0][0, 0]:.2f}
Y: {boxes_no_aug[0][0, 1]:.2f}
Z: {boxes_no_aug[0][0, 2]:.2f}
L: {boxes_no_aug[0][0, 3]:.2f}
W: {boxes_no_aug[0][0, 4]:.2f}
H: {boxes_no_aug[0][0, 5]:.2f}
Angle: {boxes_no_aug[0][0, 6]:.2f}
"""
    else:
        box_info = "No boxes"
    ax.text(0.1, 0.5, box_info, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Row 2: With augmentation
    # Camera
    ax = axes[1, 0]
    cam_img = cam_aug.permute(1, 2, 0).numpy()
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    cam_img = np.flip(cam_img, axis=0)
    ax.imshow(cam_img)
    ax.set_title("Camera (H-flipped)", fontweight='bold')
    ax.axis('off')
    
    # Heatmap
    ax = axes[1, 1]
    ax.imshow(heatmap_aug, cmap='hot', origin='lower')
    ax.set_title("Heatmap (H-flipped)", fontweight='bold')
    ax.set_xlabel("X [grid]")
    ax.set_ylabel("Y [grid]")
    
    # Box info
    ax = axes[1, 2]
    ax.axis('off')
    if boxes_aug and boxes_aug[0] is not None:
        box_info = f"""
After H-flip:
X: {boxes_aug[0][0, 0]:.2f}
Y: {boxes_aug[0][0, 1]:.2f}
Z: {boxes_aug[0][0, 2]:.2f}
L: {boxes_aug[0][0, 3]:.2f}
W: {boxes_aug[0][0, 4]:.2f}
H: {boxes_aug[0][0, 5]:.2f}
Angle: {boxes_aug[0][0, 6]:.2f}

Y_center = {y_center:.2f}
Expected Y = {y_expected:.2f}
"""
    else:
        box_info = "No boxes"
    ax.text(0.1, 0.5, box_info, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.suptitle("Augmentation Effect: H-flip Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = str(project_root / "augmentation_comparison_hflip.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    plt.close()
    
    # Now test V-flip
    print("\n" + "="*60)
    print("Testing V-flip augmentation...")
    print("="*60)
    
    dataset_vflip = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=finetune_path,
        direction="right",
        split="train",
        augment=True,
        h_flip_rate=0.0,
        v_flip_rate=1.0,  # Force V-flip
        rot_rate=0.0,
        img_mean=None,
        img_std=None,
        point_cloud_range=point_cloud_range,
    )
    
    sample_vflip = dataset_vflip[sample_idx]
    cam_vflip = sample_vflip["cam_bev"]
    heatmap_vflip = sample_vflip["targets"]["heatmap"][0].numpy()
    boxes_vflip = sample_vflip["boxes"]
    
    if boxes_vflip and boxes_vflip[0] is not None:
        print(f"  Original X: {boxes_no_aug[0][0, 0]:.2f}")
        print(f"  After V-flip X: {boxes_vflip[0][0, 0]:.2f}")
        
        # For V-flip, X should be: 2*x_center - x_original
        x_center = (point_cloud_range[0] + point_cloud_range[3]) / 2
        x_expected = 2 * x_center - boxes_no_aug[0][0, 0]
        print(f"  Expected X after V-flip: {x_expected:.2f}")
        print(f"  Difference: {abs(x_expected - boxes_vflip[0][0, 0]):.6f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: No augmentation
    ax = axes[0, 0]
    cam_img = cam_no_aug.permute(1, 2, 0).numpy()
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    cam_img = np.flip(cam_img, axis=0)
    ax.imshow(cam_img)
    ax.set_title("Camera (No Aug)", fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(heatmap_no_aug, cmap='hot', origin='lower')
    ax.set_title("Heatmap (No Aug)", fontweight='bold')
    
    ax = axes[0, 2]
    ax.axis('off')
    if boxes_no_aug and boxes_no_aug[0] is not None:
        box_info = f"""
Original (No Aug):
X: {boxes_no_aug[0][0, 0]:.2f}
Y: {boxes_no_aug[0][0, 1]:.2f}"""
    else:
        box_info = "No boxes"
    ax.text(0.1, 0.5, box_info, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Row 2: With V-flip
    ax = axes[1, 0]
    cam_img = cam_vflip.permute(1, 2, 0).numpy()
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    cam_img = np.flip(cam_img, axis=0)
    ax.imshow(cam_img)
    ax.set_title("Camera (V-flipped)", fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(heatmap_vflip, cmap='hot', origin='lower')
    ax.set_title("Heatmap (V-flipped)", fontweight='bold')
    
    ax = axes[1, 2]
    ax.axis('off')
    if boxes_vflip and boxes_vflip[0] is not None:
        box_info = f"""
After V-flip:
X: {boxes_vflip[0][0, 0]:.2f}
Y: {boxes_vflip[0][0, 1]:.2f}

X_center = {x_center:.2f}
Expected X = {x_expected:.2f}"""
    else:
        box_info = "No boxes"
    ax.text(0.1, 0.5, box_info, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.suptitle("Augmentation Effect: V-flip Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = str(project_root / "augmentation_comparison_vflip.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()
    
    # Test rotation
    print("\n" + "="*60)
    print("Testing Rotation augmentation...")
    print("="*60)
    
    dataset_rot = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=finetune_path,
        direction="right",
        split="train",
        augment=True,
        h_flip_rate=0.0,
        v_flip_rate=0.0,
        rot_rate=1.0,  # Force rotation
        rot_angle=(45, 45),  # 45 degree rotation
        img_mean=None,
        img_std=None,
        point_cloud_range=point_cloud_range,
    )
    
    sample_rot = dataset_rot[sample_idx]
    cam_rot = sample_rot["cam_bev"]
    heatmap_rot = sample_rot["targets"]["heatmap"][0].numpy()
    boxes_rot = sample_rot["boxes"]
    
    if boxes_rot and boxes_rot[0] is not None:
        print(f"  Original angle: {boxes_no_aug[0][0, 6]:.4f}")
        print(f"  After rotation: {boxes_rot[0][0, 6]:.4f}")
        print(f"  Expected change: ~0.785 rad (45 degrees)")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: No augmentation
    ax = axes[0, 0]
    cam_img = cam_no_aug.permute(1, 2, 0).numpy()
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    cam_img = np.flip(cam_img, axis=0)
    ax.imshow(cam_img)
    ax.set_title("Camera (No Aug)", fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(heatmap_no_aug, cmap='hot', origin='lower')
    ax.set_title("Heatmap (No Aug)", fontweight='bold')
    
    ax = axes[0, 2]
    ax.axis('off')
    if boxes_no_aug and boxes_no_aug[0] is not None:
        box_info = f"""
Original:
Angle: {boxes_no_aug[0][0, 6]:.4f} rad
({np.degrees(boxes_no_aug[0][0, 6]):.1f}°)"""
    else:
        box_info = "No boxes"
    ax.text(0.1, 0.5, box_info, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Row 2: With rotation
    ax = axes[1, 0]
    cam_img = cam_rot.permute(1, 2, 0).numpy()
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    cam_img = np.flip(cam_img, axis=0)
    ax.imshow(cam_img)
    ax.set_title("Camera (Rotated 45°)", fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(heatmap_rot, cmap='hot', origin='lower')
    ax.set_title("Heatmap (Rotated 45°)", fontweight='bold')
    
    ax = axes[1, 2]
    ax.axis('off')
    if boxes_rot and boxes_rot[0] is not None:
        # Compute center and angle from corner-based box
        center_x = boxes_rot[0][:, 0].mean()
        center_y = boxes_rot[0][:, 1].mean()
        orig_center_x = b_no[0][:, 0].mean()
        orig_center_y = b_no[0][:, 1].mean()
        
        box_info = f"""
After rotation (45°):
Center X: {center_x:.2f}
Center Y: {center_y:.2f}
(Original: {orig_center_x:.2f}, {orig_center_y:.2f})

Box rotated 45° around
grid center ({x_center:.1f}, {y_center:.1f})"""
    else:
        box_info = "No boxes"
    ax.text(0.1, 0.5, box_info, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.suptitle("Augmentation Effect: 45° Rotation Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = str(project_root / "augmentation_comparison_rotation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("✓ All augmentation comparison plots saved!")
    print("="*60)


if __name__ == "__main__":
    plot_augmentation_comparison()
