from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import random
from bev_multimae.preprocessing.BEV.dynamic_pillar_vfe import DynamicPillarizer, build_bev_target
import torchvision
import math
import numpy as np


def collate_finetune(batch):

    radar_list = [item['radar'] for item in batch]
    radar_batch = {}

    for k in radar_list[0].keys():
        if k == "batch_size":
            continue
        radar_batch[k] = torch.cat([r[k] for r in radar_list], dim=0)

    points_list, coords_list = [], []
    for i, r in enumerate(radar_list):
        p = r["points"].clone()
        p[:, 0] = i
        points_list.append(p)

        c = r["pillar_coords"].clone()
        c[:, 0] = i
        coords_list.append(c)

    pillar_offset = 0
    inv_list = []
    for r in radar_list:
        inv = r["pillar_inv"].clone() + pillar_offset
        inv_list.append(inv)
        pillar_offset += r["pillar_coords"].shape[0]

    radar_batch["pillar_inv"] = torch.cat(inv_list)
    radar_batch["points"] = torch.cat(points_list)
    radar_batch["pillar_coords"] = torch.cat(coords_list)
    radar_batch["batch_size"] = len(batch)

    cam_bev = torch.stack([item['cam_bev'] for item in batch])
    boxes_list = [item['boxes'] for item in batch]

    out_batch = {
        'cam_bev': cam_bev,
        'radar': radar_batch,
        'boxes': boxes_list,
    }

    if 'targets' in batch[0]:
        targets_batch = {}
        target_keys = ['heatmap', 'reg', 'height', 'dim', 'rot', 'masks']
        for key in target_keys:
            targets_batch[key] = torch.stack([item['targets'][key] for item in batch])
        out_batch['targets'] = targets_batch

    if 'img_2d' in batch[0]:
        out_batch['img_2d'] = torch.stack([item['img_2d'] for item in batch])
        out_batch['K'] = torch.stack([item['K'] for item in batch])
        out_batch['D'] = torch.stack([item['D'] for item in batch])
        out_batch['T_cam_ego'] = torch.stack([item['T_cam_ego'] for item in batch])

    return out_batch

class BEVFineData(Dataset):
    def __init__(
        self, pretrain_path, finetune_path, direction, split="train",
        img_mean=None, img_std=None, point_cloud_range=None, num_rad_channels=11,
        augment=False, v_flip_rate=0.0, h_flip_rate=0.0, rot_rate=0.0, rot_angle=(-20, 20),
        img_2d=False,
    ):
        assert split in ["train", "val", "test"]

        self.meta = torch.load(os.path.join(pretrain_path, "meta.pt"), weights_only=False)
        self.num_rad_channels = num_rad_channels
        self.img_2d = img_2d

        self.img_mean = img_mean.view(3, 1, 1) if img_mean is not None else None
        self.img_std  = img_std.view(3, 1, 1)  if img_std  is not None else None

        label_root    = Path(finetune_path) / direction / split
        pretrain_root = Path(pretrain_path) / split

        skip_events = {
            "evt_0dpi1jtdkfL4ReZx", "evt_0e3qdKh444ogAO7k",
            "evt_0e3qYFzAbTyiJEC0", "evt_0e3qZ3gUUyKSySmG",
            "evt_0e8RBM8c9lbnJIcY", "evt_0e8RGcuJoVLx95HV",
            "evt_0e8RGh9h8HVd0mMN", "evt_0e8RHpfQW2EAsyh5",
            "evt_0e8RKuQ17gJsASht", "evt_0e8RPpu7MFbAyMMZ",
            "evt_0e8RPtAGjsFkH7iI", "evt_0e8RSXbBo3Nf3CKl",
            "evt_0e8RPwdTi1LVwWTv", "evt_0e8RQ2Rhx24bzKAg",
            "evt_0e8RQLQbupv7ze9a", "evt_0e8RQQaclY2VF8AH",
            "evt_0e3qWOZPmBcxw6le", "evt_0e8RQAbtPO6JP3vW",
            "evt_0e8RQERZvAsEOyqv", "evt_0e8RQHjqh4N42EFm",
            "evt_0e8RQ72r2D5Pu6Vm", "evt_0e3qXR4PFsSC3qha"
        }

        self.samples = []

        # Cache pretrain files to avoid repeated glob/sort calls
        pretrain_files_cache = {}

        for label_file in sorted(label_root.rglob("*.npz")):
            event     = label_file.parent.name
            frame_idx = int(label_file.stem)

            if event in skip_events:
                continue

            pretrain_dir = pretrain_root / event
            if not pretrain_dir.exists():
                continue

            # Use cached pretrain files or build cache
            if event not in pretrain_files_cache:
                pretrain_files_cache[event] = sorted(
                    pretrain_dir.glob("*.pt"),
                    key=lambda p: int(p.stem.split("_")[-1])
                )

            pretrain_files = pretrain_files_cache[event]

            if frame_idx >= len(pretrain_files):
                continue

            pretrain_file = pretrain_files[frame_idx]
            img_file = pretrain_file.parent / "imgs" / f"{pretrain_file.stem}.jpg"
            cam_info_file = pretrain_file.parent / "camera_info.npz"

            # If requested, only keep samples where the original 2D image and camera info exist.
            if self.img_2d and (not img_file.exists() or not cam_info_file.exists()):
                continue

            self.samples.append((pretrain_file, label_file, img_file, cam_info_file))

        self.augment = augment
        if self.augment:
            self.pillarizer = DynamicPillarizer(
                voxel_size=self.meta["voxel_size"],
                grid_size=self.meta["grid_size"],
                point_cloud_range=point_cloud_range,
            )
            self.grid_size   = self.meta["grid_size"]
            self.rot_rate    = rot_rate
            self.h_flip_rate = h_flip_rate
            self.v_flip_rate = v_flip_rate
            self.rot_angle   = rot_angle

    def load_labels(self, path):
        data = np.load(path, allow_pickle=True)
        boxes = data["boxes"]  # object array, each entry is (8,3) or None
        boxes_list = [b for b in boxes if b is not None]

        targets = None
        target_keys = ['heatmap', 'reg', 'height', 'dim', 'rot', 'masks']
        if all(f"targets_{k}" in data for k in target_keys):
            targets_dict = {}
            for k in target_keys:
                t = torch.from_numpy(data[f"targets_{k}"]).float()
                # Permute from [H, W, C] to [C, H, W] to match PyTorch conventions
                if t.dim() == 3:
                    t = t.permute(2, 0, 1)
                targets_dict[k] = t
            targets = targets_dict

        return boxes_list, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pretrain_file, label_file, img_file, cam_info_file = self.samples[idx]

        data   = torch.load(pretrain_file, weights_only=False)
        boxes, targets = self.load_labels(label_file)

        cam = data["cam_bev"].float()

        if self.augment:
            cam, data["radar"], _ = self.augment_sample(cam, data["radar"]["points"])

        if self.img_mean is not None and self.img_std is not None:
            cam = (cam - self.img_mean) / (self.img_std + 1e-6)

        batch = {
            "cam_bev": cam,
            "radar":   data["radar"],
            "boxes":   boxes,
        }

        if targets is not None:
            batch["targets"] = targets

        if self.img_2d:
            cam_info = np.load(cam_info_file)
            batch["img_2d"] = torchvision.io.read_image(str(img_file)).float() / 255.0
            batch["K"] = torch.from_numpy(cam_info["K"]).float()
            batch["D"] = torch.from_numpy(cam_info["D"]).float()
            batch["T_cam_ego"] = torch.from_numpy(cam_info["T_cam_ego"]).float()

        return batch

    # same augment_sample as BEVDataset, minus the bev_target build
    def augment_sample(self, cam, rad_points):
        pcr      = self.pillarizer.point_cloud_range
        x_center = (pcr[0] + pcr[3]) / 2
        y_center = (pcr[1] + pcr[4]) / 2

        if self.h_flip_rate and random.random() < self.h_flip_rate:
            cam = torch.flip(cam, dims=[-2])
            rad_points[:, 2] = 2 * y_center - rad_points[:, 2]

        if self.v_flip_rate and random.random() < self.v_flip_rate:
            cam = torch.flip(cam, dims=[-1])
            rad_points[:, 1] = 2 * x_center - rad_points[:, 1]

        if self.rot_rate and random.random() < self.rot_rate:
            angle_deg = random.uniform(self.rot_angle[0], self.rot_angle[1])
            angle_rad = math.radians(angle_deg)
            cos, sin  = math.cos(angle_rad), math.sin(angle_rad)
            xy        = rad_points[:, 1:3].clone()
            cx, cy    = x_center.item(), y_center.item()

            rad_points[:, 1] = cx + cos * (xy[:, 0] - cx) - sin * (xy[:, 1] - cy)
            rad_points[:, 2] = cy + sin * (xy[:, 0] - cx) + cos * (xy[:, 1] - cy)
            cam = torchvision.transforms.functional.rotate(cam, -angle_deg)

        batch_dict = self.pillarizer.forward(rad_points)
        batch_dict["batch_size"] = 1

        return cam, batch_dict, None