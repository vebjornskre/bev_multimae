import torch
import numpy as np

def hard_splat(pts_cam, features, voxel_size, point_cloud_range, grid_size):
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    H, W = grid_size[1], grid_size[0]

    px = np.floor((pts_cam[:, 0] - x_min) / voxel_size[0]).astype(int)
    py = np.floor((pts_cam[:, 1] - y_min) / voxel_size[1]).astype(int)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, features = px[valid], py[valid], features[valid]

    bev_rgb = np.zeros((H, W, 3), dtype=np.float32)
    bev_count = np.zeros((H, W), dtype=np.float32)

    # average RGB per cell
    np.add.at(bev_rgb, (py, px), features)
    np.add.at(bev_count, (py, px), 1)

    filled = bev_count > 0
    bev_rgb[filled] /= bev_count[filled, np.newaxis]

    return torch.from_numpy(bev_rgb).permute(2, 0, 1)  # (3, H, W)


def soft_splat(pts_cam, features, depth, voxel_size, point_cloud_range, grid_size, sigma=0.5):
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    H, W = grid_size[1], grid_size[0]

    bev_rgb = np.zeros((H, W, 3), dtype=np.float32)
    bev_weight = np.zeros((H, W), dtype=np.float32)

    radius = int(np.ceil(3 * sigma / min(voxel_size[0], voxel_size[1])))

    px_f = (pts_cam[:, 0] - x_min) / voxel_size[0]
    py_f = (pts_cam[:, 1] - y_min) / voxel_size[1]

    # tighter gaussian for close points
    depth_sigma = np.clip(sigma / (depth + 1e-6), 0.1, sigma)

    # kernel offsets, all (dx, dy) within radius
    offsets = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(offsets, offsets)
    dx = dx.ravel()  # (K,)
    dy = dy.ravel()  # (K,)

    # for each kernel offset, splat all points at once
    for i in range(len(dx)):
        gx = np.floor(px_f + dx[i]).astype(int)
        gy = np.floor(py_f + dy[i]).astype(int)

        valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        gx, gy = gx[valid], gy[valid]

        w = np.exp(-0.5 * (dx[i]**2 + dy[i]**2) / depth_sigma[valid]**2)

        np.add.at(bev_rgb, (gy, gx), w[:, None] * features[valid])
        np.add.at(bev_weight, (gy, gx), w)

    filled = bev_weight > 0
    bev_rgb[filled] /= bev_weight[filled, np.newaxis]

    return torch.from_numpy(bev_rgb).permute(2, 0, 1)  # (3, H, W)

def patchify(bev, patch_size):
    C, H, W = bev.shape
    assert H % patch_size == 0 and W % patch_size == 0
    
    Ph = H // patch_size
    Pw = W // patch_size
    
    # (C, H, W) -> (Ph*Pw, C*patch_size*patch_size)
    patches = bev.reshape(C, Ph, patch_size, Pw, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4)  # (Ph, Pw, C, patch_size, patch_size)
    patches = patches.reshape(Ph * Pw, C * patch_size * patch_size)
    
    return patches