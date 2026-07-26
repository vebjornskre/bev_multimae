import copy
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from bev_multimae.datasets.finetuning_data import BEVFineData


def _bev_canvas(cam_bev):
    img = cam_bev.detach().cpu()

    if img.dim() == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    elif img.dim() == 3:
        img = img[:3] if img.shape[0] >= 3 else img.repeat(3, 1, 1)
    elif img.dim() == 4:
        img = img[0, :3] if img.shape[1] >= 3 else img[0].repeat(3, 1, 1)

    img = img.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


def _project_pts(xy_world, pcr, h, w):
    x0, y0, _, x1, y1, _ = pcr

    px = (xy_world[:, 0] - x0) / (x1 - x0) * w
    py = (xy_world[:, 1] - y0) / (y1 - y0) * h

    return np.stack([px, py], axis=1)


def _draw_panel(ax, img, boxes, radar, pcr, title):
    h, w = img.shape[:2]

    ax.imshow(img, origin="upper", aspect="auto")

    radar_px = _project_pts(radar, pcr, h, w)
    ax.scatter(
        radar_px[:, 0],
        radar_px[:, 1],
        s=6,
        c="lime",
        alpha=0.8,
        linewidths=0,
    )

    for box in boxes:
        corners = _project_pts(box[:4, :2], pcr, h, w)

        ax.add_patch(
            plt.Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor="#00FF99",
                linewidth=1.5,
            )
        )

    ax.set_title(title, fontsize=10, color="white", pad=6)
    ax.axis("off")


def visualize_augmentations(cfg, sample_idx=0, save_dir=None, angle_deg=30):
    direction = cfg.get("direction", "right")
    split = cfg.get("split", "val")

    pretrain_path = (
        cfg.processed_data_dir_right
        if direction == "right"
        else cfg.processed_data_dir_left
    )

    pcr = (
        cfg.right_point_cloud_range
        if direction == "right"
        else cfg.left_point_cloud_range
    )

    try:
        stats = torch.load(
            os.path.join(cfg.processed_data_dir, "mean_std.pt"),
            map_location="cpu",
        )
        img_mean = stats["img_mean"]
        img_std = stats["img_std"]
    except Exception:
        img_mean = None
        img_std = None

    ds = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=cfg.finetuning_data_dir,
        direction=direction,
        split=split,
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=pcr,
        augment=True,
        h_flip_rate=1.0,
        v_flip_rate=1.0,
        rot_rate=1.0,
        rot_angle=(angle_deg, angle_deg),
        img_2d=False,
    )

    pretrain_file, label_file, _, _ = ds.samples[sample_idx]

    raw = torch.load(
        pretrain_file,
        map_location="cpu",
        weights_only=False,
    )

    boxes, _ = ds.load_labels(label_file)

    cam = raw["cam_bev"].float()
    bev_feat = raw["bev_feat"].float() if "bev_feat" in raw else None
    radar = raw["radar"]["points"].clone()

    print("cam shape:", cam.shape)
    print("radar points shape:", radar.shape)
    print("num boxes:", len(boxes))
    print("pcr:", pcr)

    pcr_list = list(pcr)

    def h_flip(cam, bev_feat, radar, boxes):
        x_center = (pcr[0] + pcr[3]) / 2

        cam = torch.flip(cam, dims=[-1])

        if bev_feat is not None:
            bev_feat = torch.flip(bev_feat, dims=[-1])

        radar = radar.clone()
        radar[:, 1] = 2 * x_center - radar[:, 1]

        boxes = ds._transform_boxes_x_flip(
            boxes,
            x_center,
        )

        return cam, bev_feat, radar, boxes

    def v_flip(cam, bev_feat, radar, boxes):
        y_center = (pcr[1] + pcr[4]) / 2

        cam = torch.flip(cam, dims=[-2])

        if bev_feat is not None:
            bev_feat = torch.flip(bev_feat, dims=[-2])

        radar = radar.clone()
        radar[:, 2] = 2 * y_center - radar[:, 2]

        boxes = ds._transform_boxes_y_flip(
            boxes,
            y_center,
        )

        return cam, bev_feat, radar, boxes

    def rotate(cam, bev_feat, radar, boxes):
        import torchvision

        x_center = float((pcr[0] + pcr[3]) / 2)
        y_center = float((pcr[1] + pcr[4]) / 2)

        angle_rad = math.radians(angle_deg)
        cos = math.cos(angle_rad)
        sin = math.sin(angle_rad)

        radar = radar.clone()
        xy = radar[:, 1:3].clone()

        radar[:, 1] = (
            x_center
            + cos * (xy[:, 0] - x_center)
            - sin * (xy[:, 1] - y_center)
        )

        radar[:, 2] = (
            y_center
            + sin * (xy[:, 0] - x_center)
            + cos * (xy[:, 1] - y_center)
        )

        cam = torchvision.transforms.functional.rotate(
            cam,
            -angle_deg,
        )

        if bev_feat is not None:
            bev_feat = torchvision.transforms.functional.rotate(
                bev_feat,
                -angle_deg,
            )

        boxes = ds._transform_boxes_rotate(
            boxes,
            angle_rad,
            x_center,
            y_center,
        )

        return cam, bev_feat, radar, boxes

    variants = [
        ("original", cam, bev_feat, radar, boxes),
    ]

    c, f, r, b = h_flip(
        cam.clone(),
        copy.deepcopy(bev_feat),
        radar.clone(),
        copy.deepcopy(boxes),
    )
    variants.append(("h-flip", c, f, r, b))

    c, f, r, b = v_flip(
        cam.clone(),
        copy.deepcopy(bev_feat),
        radar.clone(),
        copy.deepcopy(boxes),
    )
    variants.append(("v-flip", c, f, r, b))

    c, f, r, b = rotate(
        cam.clone(),
        copy.deepcopy(bev_feat),
        radar.clone(),
        copy.deepcopy(boxes),
    )
    variants.append((f"rotate {angle_deg}°", c, f, r, b))

    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(5 * len(variants), 5),
    )

    fig.patch.set_facecolor("#111111")

    for ax, (name, c, _, r, b) in zip(axes, variants):
        ax.set_facecolor("#111111")

        boxes_np = [
            box.detach().cpu().numpy()
            if torch.is_tensor(box)
            else box
            for box in b
            if box is not None
        ]

        radar_np = r.detach().cpu().numpy()[:, 1:3]

        _draw_panel(
            ax,
            _bev_canvas(c),
            boxes_np,
            radar_np,
            pcr_list,
            name,
        )

    fig.legend(
        handles=[
            mpatches.Patch(
                edgecolor="#00FF99",
                facecolor="none",
                label="GT boxes",
            ),
            mpatches.Patch(
                color="#FF4466",
                label="Radar points",
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=9,
        framealpha=0.3,
        facecolor="#222222",
        edgecolor="none",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Augmentation check — sample {sample_idx}",
        color="white",
        fontsize=12,
        y=1.01,
    )

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        out = os.path.join(
            save_dir,
            f"augmentation_sample_{sample_idx}.png",
        )

        plt.savefig(
            out,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

        print("Saved to:", out)
    else:
        plt.show()

    plt.close()