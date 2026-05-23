import numpy as np
import torch
import math
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os


def corners_to_center_format(corners_8x3):
    """
    Convert 8-corner box format to center-based format.

    Args:
        corners_8x3: numpy array (8, 3) - 8 corners of the box

    Returns:
        dict with keys: 'center', 'size', 'angle'
        - center: (x, y, z)
        - size: (length, width, height)
        - angle: rotation angle in radians (0 for now, as we don't have orientation)
    """
    if corners_8x3 is None or len(corners_8x3) == 0:
        return None

    corners = np.array(corners_8x3, dtype=np.float32)

    # Extract center (mean of all corners)
    center = corners.mean(axis=0)

    # Extract size: compute edge lengths
    # Typically: corners are ordered such that:
    # [0-3] are bottom face, [4-7] are top face
    # Or: sort by z coordinate to separate bottom and top
    bottom_corners = corners[corners[:, 2] < corners[:, 2].mean()]
    top_corners = corners[corners[:, 2] >= corners[:, 2].mean()]

    # Compute dimensions from corners
    # Length (x), Width (y), Height (z)
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    z_coords = corners[:, 2]

    length = x_coords.max() - x_coords.min()
    width = y_coords.max() - y_coords.min()
    height = z_coords.max() - z_coords.min()

    # Angle: For now, assume 0 (can be refined if box orientation is stored)
    angle = 0.0

    return {
        'center': center,  # (3,)
        'size': np.array([length, width, height], dtype=np.float32),  # (3,)
        'angle': angle,  # scalar
    }

def gaussian_2d(shape, sigma=1.0):
    """Create a 2D Gaussian kernel."""
    x = np.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2 + 1)
    y = np.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2 + 1)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return Z / Z.max()


def build_centerpoint_targets_with_gaussian_gpu(
    boxes_list,
    bev_range=(-20, -20, 20, 20),
    grid_size=64,
    point_cloud_range=None,
    gaussian_radius=2,
    device="cuda",
):
    if point_cloud_range is None:
        x_min, y_min, x_max, y_max = bev_range
    else:
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    heatmap = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32)
    reg = torch.zeros(grid_size, grid_size, 2, dtype=torch.float32)
    height = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32)
    dim = torch.zeros(grid_size, grid_size, 3, dtype=torch.float32)
    rot = torch.zeros(grid_size, grid_size, 2, dtype=torch.float32)
    masks = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32)

    g = torch.from_numpy(
        gaussian_2d(
            (gaussian_radius * 2 + 1, gaussian_radius * 2 + 1),
            sigma=gaussian_radius / 3,
        )
    ).float()

    for box in boxes_list:
        if box is None:
            continue

        c = corners_to_center_format(box)
        if c is None:
            continue

        center = c["center"]
        size = c["size"]
        angle = float(c["angle"])

        x_idx = float((center[0] - x_min) / (x_max - x_min) * grid_size)
        y_idx = float((center[1] - y_min) / (y_max - y_min) * grid_size)

        if x_idx < 0 or x_idx >= grid_size or y_idx < 0 or y_idx >= grid_size:
            continue
    
        x_int = int(x_idx)
        y_int = int(y_idx)

        x0 = max(0, x_int - gaussian_radius)
        x1 = min(grid_size, x_int + gaussian_radius + 1)
        y0 = max(0, y_int - gaussian_radius)
        y1 = min(grid_size, y_int + gaussian_radius + 1)

        gx0 = x0 - (x_int - gaussian_radius)
        gx1 = gx0 + (x1 - x0)
        gy0 = y0 - (y_int - gaussian_radius)
        gy1 = gy0 + (y1 - y0)

        heatmap[y0:y1, x0:x1, 0] = torch.maximum(
            heatmap[y0:y1, x0:x1, 0],
            g[gy0:gy1, gx0:gx1],
        )

        reg[y_int, x_int, 0] = float(x_idx - x_int)
        reg[y_int, x_int, 1] = float(y_idx - y_int)

        height[y_int, x_int, 0] = float(center[2])

        dim[y_int, x_int, 0] = float(size[0])
        dim[y_int, x_int, 1] = float(size[1])
        dim[y_int, x_int, 2] = float(size[2])

        rot[y_int, x_int, 0] = math.sin(angle)
        rot[y_int, x_int, 1] = math.cos(angle)

        masks[y_int, x_int, 0] = 1.0

    return {
        "heatmap": heatmap,
        "reg": reg,
        "height": height,
        "dim": dim,
        "rot": rot,
        "masks": masks,
    }


def visualize_targets(targets, boxes_list, point_cloud_range, filename, title="Target Visualization"):
    """Visualize generated targets as subplots.

    Args:
        targets: dict with keys 'heatmap', 'reg', 'height', 'dim', 'rot', 'masks'
        boxes_list: list of (8, 3) box corners for overlay
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
        filename: path to save visualization
        title: plot title
    """
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    grid_size = targets['heatmap'].shape[0]

    # Convert tensors to numpy
    heatmap_np = targets['heatmap'].squeeze().cpu().numpy()
    reg_np = targets['reg'].cpu().numpy()
    height_np = targets['height'].squeeze().cpu().numpy()
    dim_np = targets['dim'].cpu().numpy()
    rot_np = targets['rot'].cpu().numpy()
    masks_np = targets['masks'].squeeze().cpu().numpy()

    # Compute reg magnitude for visualization
    reg_mag = np.sqrt(reg_np[:, :, 0]**2 + reg_np[:, :, 1]**2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # Heatmap
    ax = axes[0, 0]
    im = ax.imshow(heatmap_np, origin='lower', cmap='hot')
    ax.set_title('Heatmap')
    plt.colorbar(im, ax=ax)

    # Overlay box centers as green dots
    for box in boxes_list:
        if box is not None:
            center = box.mean(axis=0)
            x_idx = (center[0] - x_min) / (x_max - x_min) * grid_size
            y_idx = (center[1] - y_min) / (y_max - y_min) * grid_size
            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                ax.plot(x_idx, y_idx, 'g+', markersize=12, markeredgewidth=2)

    ax.set_ylabel('Y (lateral)')
    ax.set_xlabel('X (forward)')

    # Regression magnitude
    ax = axes[0, 1]
    im = ax.imshow(reg_mag, origin='lower', cmap='viridis')
    ax.set_title('Regression Magnitude')
    plt.colorbar(im, ax=ax)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    # Height
    ax = axes[0, 2]
    im = ax.imshow(height_np, origin='lower', cmap='plasma')
    ax.set_title('Height (Z)')
    plt.colorbar(im, ax=ax)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    # Dimensions (average)
    dim_avg = (dim_np[:, :, 0] + dim_np[:, :, 1] + dim_np[:, :, 2]) / 3
    ax = axes[1, 0]
    im = ax.imshow(dim_avg, origin='lower', cmap='cool')
    ax.set_title('Dimensions (avg L/W/H)')
    plt.colorbar(im, ax=ax)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    # Rotation magnitude
    rot_mag = np.sqrt(rot_np[:, :, 0]**2 + rot_np[:, :, 1]**2)
    ax = axes[1, 1]
    im = ax.imshow(rot_mag, origin='lower', cmap='twilight')
    ax.set_title('Rotation Magnitude')
    plt.colorbar(im, ax=ax)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    # Mask
    ax = axes[1, 2]
    im = ax.imshow(masks_np, origin='lower', cmap='binary')
    ax.set_title(f'Mask (valid locations, count: {masks_np.sum():.0f})')
    plt.colorbar(im, ax=ax)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved visualization to {filename}")