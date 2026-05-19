import numpy as np
import torch
import math
from scipy.spatial import ConvexHull


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


def build_centerpoint_targets(boxes_list, bev_range=(-20, -20, 20, 20), grid_size=128, point_cloud_range=None):
    """
    Build CenterPoint-style targets from list of boxes.

    Args:
        boxes_list: list of (8, 3) corner arrays or None entries
        bev_range: (x_min, y_min, x_max, y_max) in world coordinates
        grid_size: output grid size (128x128)
        point_cloud_range: if provided, use this instead of bev_range

    Returns:
        targets dict:
        - heatmap: (grid_size, grid_size, 1) with Gaussian at center locations
        - reg: (grid_size, grid_size, 2) with xy offset
        - height: (grid_size, grid_size, 1) with z coordinate
        - dim: (grid_size, grid_size, 3) with lwh
        - rot: (grid_size, grid_size, 2) with [sin(yaw), cos(yaw)]
        - masks: (grid_size, grid_size, 1) indicating valid locations
    """

    if point_cloud_range is None:
        x_min, y_min, x_max, y_max = bev_range
    else:
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    # Initialize target tensors
    heatmap = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32)
    reg = torch.zeros(grid_size, grid_size, 2, dtype=torch.float32)
    height = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32)
    dim = torch.zeros(grid_size, grid_size, 3, dtype=torch.float32)
    rot = torch.zeros(grid_size, grid_size, 2, dtype=torch.float32)
    masks = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32)

    # Process each box
    for box in boxes_list:
        if box is None:
            continue

        center_dict = corners_to_center_format(box)
        if center_dict is None:
            continue

        center = center_dict['center']
        size = center_dict['size']
        angle = center_dict['angle']

        # Project center to grid coordinates
        x_idx = (center[0] - x_min) / (x_max - x_min) * grid_size
        y_idx = (center[1] - y_min) / (y_max - y_min) * grid_size

        # Skip if out of bounds
        if x_idx < 0 or x_idx >= grid_size or y_idx < 0 or y_idx >= grid_size:
            continue

        x_idx_int = int(x_idx)
        y_idx_int = int(y_idx)

        # Clamp to valid range
        x_idx_int = max(0, min(x_idx_int, grid_size - 1))
        y_idx_int = max(0, min(y_idx_int, grid_size - 1))

        # Set heatmap (Gaussian blob around center)
        heatmap[y_idx_int, x_idx_int, 0] = 1.0

        # Set regression targets (offset from grid point)
        reg[y_idx_int, x_idx_int, 0] = torch.tensor(x_idx - x_idx_int, dtype=torch.float32)
        reg[y_idx_int, x_idx_int, 1] = torch.tensor(y_idx - y_idx_int, dtype=torch.float32)

        # Set height (z coordinate)
        height[y_idx_int, x_idx_int, 0] = torch.tensor(center[2], dtype=torch.float32)

        # Set dimensions
        dim[y_idx_int, x_idx_int, 0] = torch.tensor(size[0], dtype=torch.float32)  # length
        dim[y_idx_int, x_idx_int, 1] = torch.tensor(size[1], dtype=torch.float32)  # width
        dim[y_idx_int, x_idx_int, 2] = torch.tensor(size[2], dtype=torch.float32)  # height

        # Set rotation (as sin, cos)
        rot[y_idx_int, x_idx_int, 0] = torch.tensor(np.sin(angle), dtype=torch.float32)
        rot[y_idx_int, x_idx_int, 1] = torch.tensor(np.cos(angle), dtype=torch.float32)

        # Set mask
        masks[y_idx_int, x_idx_int, 0] = 1.0

    return {
        'heatmap': heatmap,
        'reg': reg,
        'height': height,
        'dim': dim,
        'rot': rot,
        'masks': masks,
    }


def gaussian_2d(shape, sigma=1.0):
    """Create a 2D Gaussian kernel."""
    x = np.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2 + 1)
    y = np.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2 + 1)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return Z / Z.max()


def build_centerpoint_targets_with_gaussian_gpu(boxes_list, bev_range=(-20, -20, 20, 20), grid_size=64, point_cloud_range=None, gaussian_radius=2, device='cuda'):
    """
    GPU version: Build CenterPoint-style targets with Gaussian heatmaps on GPU.
    Optimized for minimal tensor creation overhead.

    Args:
        boxes_list: list of (8, 3) arrays (can be numpy or torch)
        bev_range: (x_min, y_min, x_max, y_max) in world coordinates
        grid_size: output grid size (64x64)
        device: 'cuda' or 'cpu'

    Returns:
        targets dict with all tensors on specified device
    """
    if point_cloud_range is None:
        x_min, y_min, x_max, y_max = bev_range
    else:
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    # Initialize target tensors on GPU
    heatmap = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32, device=device)
    reg = torch.zeros(grid_size, grid_size, 2, dtype=torch.float32, device=device)
    height = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32, device=device)
    dim = torch.zeros(grid_size, grid_size, 3, dtype=torch.float32, device=device)
    rot = torch.zeros(grid_size, grid_size, 2, dtype=torch.float32, device=device)
    masks = torch.zeros(grid_size, grid_size, 1, dtype=torch.float32, device=device)

    # Create Gaussian kernel on GPU
    gaussian_kernel = torch.from_numpy(
        gaussian_2d((gaussian_radius * 2 + 1, gaussian_radius * 2 + 1), sigma=gaussian_radius / 3)
    ).to(device)

    # Process each box
    for box in boxes_list:
        if box is None:
            continue

        center_dict = corners_to_center_format(box)
        if center_dict is None:
            continue

        center = torch.tensor(center_dict['center'], dtype=torch.float32, device=device)
        size = torch.tensor(center_dict['size'], dtype=torch.float32, device=device)
        angle = center_dict['angle']

        # Project center to grid coordinates
        x_idx = (center[0] - x_min) / (x_max - x_min) * grid_size
        y_idx = (center[1] - y_min) / (y_max - y_min) * grid_size

        # Skip if out of bounds
        if x_idx < 0 or x_idx >= grid_size or y_idx < 0 or y_idx >= grid_size:
            continue

        x_idx_int = int(x_idx.item())
        y_idx_int = int(y_idx.item())

        # Add Gaussian blob around center using optimized nested loops
        for dx in range(-gaussian_radius, gaussian_radius + 1):
            for dy in range(-gaussian_radius, gaussian_radius + 1):
                px = x_idx_int + dx
                py = y_idx_int + dy

                if 0 <= px < grid_size and 0 <= py < grid_size:
                    gx = dx + gaussian_radius
                    gy = dy + gaussian_radius
                    heatmap[py, px, 0] = torch.max(heatmap[py, px, 0], gaussian_kernel[gy, gx])

        # Set regression targets (offset from integer grid point)
        reg[y_idx_int, x_idx_int, 0] = x_idx - x_idx_int
        reg[y_idx_int, x_idx_int, 1] = y_idx - y_idx_int

        # Set height
        height[y_idx_int, x_idx_int, 0] = center[2]

        # Set dimensions
        dim[y_idx_int, x_idx_int, 0] = size[0]
        dim[y_idx_int, x_idx_int, 1] = size[1]
        dim[y_idx_int, x_idx_int, 2] = size[2]

        # Set rotation using numpy (faster than tensor creation)
        rot[y_idx_int, x_idx_int, 0] = np.sin(angle)
        rot[y_idx_int, x_idx_int, 1] = np.cos(angle)

        # Set mask
        masks[y_idx_int, x_idx_int, 0] = 1.0

    return {
        'heatmap': heatmap,
        'reg': reg,
        'height': height,
        'dim': dim,
        'rot': rot,
        'masks': masks,
    }
