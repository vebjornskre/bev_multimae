import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d
import matplotlib.patches as patches

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

def scatter_seg_points(segs_pts, segs_colors, center_sphere, path, save_ply=True):
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