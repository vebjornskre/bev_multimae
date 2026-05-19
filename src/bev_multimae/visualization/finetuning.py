import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d
import matplotlib.patches as patches
import math
import os
import torch

from bev_multimae.preprocessing.camera.lift import lift
from bev_multimae.preprocessing.sync import load_radar, load_seg
from bev_multimae.preprocessing.radar.radar_process_utils import radar_to_ego
from bev_multimae.preprocessing.BEV.splat import hard_splat, patchify
from bev_multimae.visualization.BEV_visualization import overlay_radar_on_image, plot_bev_comparison
from bev_multimae.preprocessing.get_transforms import apply_transform

def _merge_radar(paths: list) -> dict:
        frames = [load_radar(p) for p in paths]
        return {k: np.concatenate([f[k] for f in frames]) for k in frames[0].keys()}

def _compute_grid_size(voxel_size, point_cloud_range) -> list:
    pcr = point_cloud_range
    return [
        math.ceil((pcr[3] - pcr[0]) / voxel_size[0]),  # X (forward) → nx
        math.ceil((pcr[4] - pcr[1]) / voxel_size[1]),  # Y (lateral) → ny
        math.ceil((pcr[5] - pcr[2]) / voxel_size[2]),  # Z → nz
    ]

def plot_img_and_seg(img, mask, bboxes, path):
    img = np.array(img)
    if img is None or mask is None:
        print("Missing image or mask")
        return

    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Segmentation")
    axes[2].imshow(blended)
    axes[2].set_title("Overlay + bboxes")

    for box in bboxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[2].add_patch(rect)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def scatter_seg_points(segs_pts, segs_colors, center_sphere, path, save_ply=False):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    combined_pcd = o3d.geometry.PointCloud()

    for seg_pts, seg_colors in zip(segs_pts, segs_colors):
        seg_pts = np.asarray(seg_pts)
        seg_colors = np.asarray(seg_colors)

        if seg_colors.max() > 1:
            seg_colors = seg_colors / 255.0

        valid = np.isfinite(seg_pts).all(axis=1)
        seg_pts = seg_pts[valid]
        seg_colors = seg_colors[valid]

        ax.scatter(seg_pts[:, 0], seg_pts[:, 1], seg_pts[:, 2], c=seg_colors, s=1)

        if save_ply:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg_pts)
            pcd.colors = o3d.utility.Vector3dVector(seg_colors)
            combined_pcd += pcd

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    if save_ply:
        combined_pcd += center_sphere
        o3d.io.write_point_cloud(path.replace(".png", ".ply"), combined_pcd)

def make_sphere_pts(centers, radius=0.5, n=1000, color=[0, 0, 1]):
    centers = np.atleast_2d(centers)
    pcd = o3d.geometry.PointCloud()
    for center in centers:
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radius + center
        colors = np.tile(color, (n, 1)).astype(np.float64)
        sphere = o3d.geometry.PointCloud()
        sphere.points = o3d.utility.Vector3dVector(pts)
        sphere.colors = o3d.utility.Vector3dVector(colors)
        pcd += sphere
    return pcd

BOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

def plot_boxes_on_img(img, boxes, K, D, T_cam_ego, save_path):
    img_np = np.array(img).copy()
    T_ego_cam = np.linalg.inv(T_cam_ego)

    for j, box in enumerate(boxes):
        if box is None:
            continue

        box_cam = apply_transform(T_ego_cam, box)

        if np.any(box_cam[:, 2] <= 0):
            continue

        color = BOX_COLORS[j % len(BOX_COLORS)]
        center = box.mean(axis=0)

        pts, _ = cv2.projectPoints(
            box_cam.astype(np.float64),
            np.zeros(3), np.zeros(3),
            K.astype(np.float64), D,
        )
        pts = pts.reshape(-1, 2).astype(int)

        n = len(pts)
        half = n // 2

        for face in [range(half), range(half, n)]:
            face = list(face)
            for i in range(len(face)):
                cv2.line(img_np, tuple(pts[face[i]]), tuple(pts[face[(i + 1) % len(face)]]), color, 2)

        for i in range(half):
            cv2.line(img_np, tuple(pts[i]), tuple(pts[i + half]), color, 2)

        label = f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
        top = pts[:, 1].argmin()
        cv2.putText(img_np, label, tuple(pts[top]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def plot_boxes_topdown(boxes, bev, voxel_size, point_cloud_range, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    if isinstance(bev, torch.Tensor):
        bev = bev.numpy()
    if bev.shape[0] == 3:
        bev = np.transpose(bev, (1, 2, 0))
    bev = np.clip(bev, 0, 1)
    ax.imshow(bev, origin='lower')

    colors = ['g', 'r', 'b', 'y', 'm', 'c']
    pcr = point_cloud_range

    def ego_to_bev_px(x, y):
        row = (y - pcr[1]) / voxel_size[1]
        col = (x - pcr[0]) / voxel_size[0]
        return row, col

    for j, box in enumerate(boxes):
        if box is None:
            continue

        color = BOX_COLORS[j % len(BOX_COLORS)]
        color = (color[0] / 255, color[1] / 255, color[2] / 255)
        center = box.mean(axis=0)

        top_idx = np.argsort(box[:, 2])[-4:]
        bottom_idx = np.argsort(box[:, 2])[:4]

        footprint_pts = box[bottom_idx, :2]
        x_min, x_max = footprint_pts[:, 0].min(), footprint_pts[:, 0].max()
        y_min, y_max = footprint_pts[:, 1].min(), footprint_pts[:, 1].max()

        footprint = np.array([
            [x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max],
            [x_min, y_min],
        ])

        rows, cols = zip(*[ego_to_bev_px(p[0], p[1]) for p in footprint])
        ax.plot(cols, rows, color=color, linewidth=1.5)

        cr, cc = ego_to_bev_px(center[0], center[1])
        ax.plot(cc, cr, 'x', color=color)
        ax.text(cc, cr, f"({center[0]:.1f}, {center[1]:.1f})", fontsize=7, color=color)

    ax.set_ylabel("Y (lateral)")
    ax.set_xlabel("X (forward)")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_human_boxes(
    cfg,
    event,
    frame,
    img,
    lidar,
    bboxes,
    boxes,
    centers,
    depth_model,
    T_cam_ego,
    T_rad_ego,
    T_rad_cam,
    T_lid_cam,
    K,
    D
):

    save_dir = os.path.join(cfg.finetuning_vis, cfg.direction, event)
    os.makedirs(save_dir, exist_ok=True)

    radar = _merge_radar(frame["rad"])
    seg = load_seg(frame["seg"])

    ego_radar = radar_to_ego(cfg, radar, T_rad_ego)

    pts_cam_ego, colors, segs_pts, segs_colors = lift(
        cfg,
        img,
        lidar,
        depth_model,
        T_cam_ego,
        T_lid_cam,
        T_rad_cam,
        plot=False,
        seg_mask=seg,
        bboxes=bboxes,
    )

    valid_centers = np.array([c for c in centers if c is not None])

    if len(valid_centers) > 0:
        center_spheres = make_sphere_pts(valid_centers, radius=0.15)
        pts_cam_ego = np.concatenate([pts_cam_ego, np.asarray(center_spheres.points)], axis=0)
        colors = np.concatenate([colors, np.asarray(center_spheres.colors)], axis=0)

    for box in boxes:
        if box is None:
            continue

        box_spheres = make_sphere_pts(box, radius=0.08, color=[1, 0, 0])
        pts_cam_ego = np.concatenate([pts_cam_ego, np.asarray(box_spheres.points)], axis=0)
        colors = np.concatenate([colors, np.asarray(box_spheres.colors)], axis=0)

    voxel_size = cfg.voxel_size
    hi_res_voxel = cfg.hi_res_voxel
    patch_size_pixels = int(voxel_size[0] / hi_res_voxel[0])

    if cfg.direction == "right":
        point_cloud_range = cfg.right_point_cloud_range
    elif cfg.direction == "left":
        point_cloud_range = cfg.left_point_cloud_range

    hi_res_grid_size = _compute_grid_size(hi_res_voxel, point_cloud_range)

    bev_cam_hires = hard_splat(
        pts_cam_ego,
        colors,
        hi_res_voxel,
        point_cloud_range,
        hi_res_grid_size,
    )

    img_with_radar = overlay_radar_on_image(cfg, img, ego_radar, T_cam_ego)

    plot_bev_comparison(
        cfg,
        img_with_radar,
        ego_radar,
        bev_cam_hires,
        voxel_size,
        point_cloud_range,
        patch_size_pixels,
        "test",
        0,
        manual_save=save_dir,
    )

    scatter_seg_points(
        segs_pts,
        segs_colors,
        None,
        os.path.join(save_dir, "seg_pts.png"),
    )

    plot_img_and_seg(
        img,
        seg,
        bboxes,
        os.path.join(save_dir, "img_n_seg.png"),
    )
    save_path = os.path.join(save_dir, '3d_box_plot.png')
    plot_boxes_on_img(img, boxes, K, D, T_cam_ego, save_path)
    plot_boxes_topdown(
        boxes, bev_cam_hires, hi_res_voxel, point_cloud_range,
        os.path.join(save_dir, "boxes_topdown.png")
    )