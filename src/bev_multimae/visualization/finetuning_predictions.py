import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2

def norm_img(x):
    x = x.detach().cpu().float()
    x = x.permute(1, 2, 0).numpy()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def save_detections(detections, targets, batch, folder, pred_boxes=None, target_boxes=None, point_cloud_range=None, voxel_size=None):
    os.makedirs(folder, exist_ok=True)

    inp = norm_img(batch["cam_bev"][0])
    heatmap = torch.sigmoid(detections["heatmap"][0, 0]).detach().cpu().numpy()
    cmap = "inferno"

    if targets is not None and "heatmap" in targets:
        target_heatmap = targets["heatmap"][0, 0].detach().cpu().numpy()
        vmax = max(0.1, heatmap.max(), target_heatmap.max())
    else:
        target_heatmap = None
        vmax = max(0.1, heatmap.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(inp, origin="lower")
    axes[0].set_title("Input")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Heatmap probability")

    plt.savefig(os.path.join(folder, "input_prediction.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()

    if target_heatmap is not None:
        fig, axes = plt.subplots(1, 3, figsize=(26, 8))

        axes[0].imshow(inp, origin="lower")
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        im = axes[2].imshow(target_heatmap, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        axes[2].set_title("Target")
        axes[2].axis("off")

        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Heatmap probability")

        plt.savefig(os.path.join(folder, "input_prediction_target.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
        plt.close()

    if pred_boxes is not None or target_boxes is not None:
        bev = norm_img(batch["cam_bev"][0])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(bev, origin="lower")

        # overlay radar points
        if "radar" in batch:
            pts = batch["radar"]["points"]
            if torch.is_tensor(pts):
                pts = pts.detach().cpu().numpy()
            # points columns: [batch_idx, x, y, z, ...]
            mask = pts[:, 0] == 0
            rx = pts[mask, 1]
            ry = pts[mask, 2]
            col = (rx - point_cloud_range[0]) / voxel_size[0]
            row = (ry - point_cloud_range[1]) / voxel_size[1]
            ax.scatter(col, row, s=8, c="lime", linewidths=0)

        def draw_box(box, color, label):
            center = box.mean(axis=0)
            bottom = box[np.argsort(box[:, 2])[:4], :2]

            x_min, x_max = bottom[:, 0].min(), bottom[:, 0].max()
            y_min, y_max = bottom[:, 1].min(), bottom[:, 1].max()

            footprint = [
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max],
                [x_min, y_min],
            ]

            rows = [(p[1] - point_cloud_range[1]) / voxel_size[1] for p in footprint]
            cols = [(p[0] - point_cloud_range[0]) / voxel_size[0] for p in footprint]

            ax.plot(cols, rows, color=color, linewidth=2)

            cc = np.mean(cols)
            cr = max(rows) + 2

            ax.text(
                cc,
                cr,
                f"{label} ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})",
                fontsize=7,
                color=color,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
            )

        text_i = 0

        if pred_boxes is not None:
            for box in pred_boxes:
                draw_box(box, "lime", "pred")

        if target_boxes is not None:
            for box in target_boxes:
                draw_box(box, "red", "target")

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "pred_target_boxes_bev.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
        plt.close()


def plot_boxes_on_image(img_tensor, pred_boxes, target_boxes, K, D, save_path):
    """Project 3D boxes onto 2D image and save."""
    img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8).copy()

    K_np = K[0].cpu().numpy().astype(np.float64)
    D_np = D[0].cpu().numpy().astype(np.float64)

    FACES = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [1,2,6,5], [0,3,7,4]]

    def project_box(corners, color):
        pts, _ = cv2.projectPoints(
            corners.astype(np.float64),
            np.zeros(3), np.zeros(3), K_np, D_np
        )
        pts = pts.reshape(-1, 2).astype(int)

        if np.any(corners[:, 2] <= 0):  # behind camera
            return

        for face in FACES:
            for i in range(len(face)):
                cv2.line(img, tuple(pts[face[i]]), tuple(pts[face[(i+1) % len(face)]]), color, 2)

    for box in target_boxes:
        project_box(box, color=(255, 0, 0))  # red

    for box in pred_boxes:
        project_box(box, color=(0, 255, 0))  # green

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))