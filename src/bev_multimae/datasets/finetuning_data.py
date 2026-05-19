from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import random
from bev_multimae.preprocessing.BEV.dynamic_pillar_vfe import DynamicPillarizer, build_bev_target
import torchvision
import math

class BEVFineData(Dataset):
    def __init__(
        self, pretrain_path, finetune_path, direction, split="train",
        img_mean=None, img_std=None, point_cloud_range=None, num_rad_channels=11,
        augment=False, v_flip_rate=0.0, h_flip_rate=0.0, rot_rate=0.0, rot_angle=(-20, 20),
    ):
        assert split in ["train", "val", "test"]

        self.meta = torch.load(os.path.join(pretrain_path, "meta.pt"), weights_only=False)
        self.num_rad_channels = num_rad_channels

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
        for label_file in sorted(label_root.rglob("*.npz")):
            event     = label_file.parent.name
            frame_idx = int(label_file.stem)

            if event in skip_events:
                continue

            pretrain_dir = pretrain_root / event
            if not pretrain_dir.exists():
                continue

            pretrain_files = sorted(pretrain_dir.glob("*.pt"), key=lambda p: int(p.stem.split("_")[-1]))

            if frame_idx >= len(pretrain_files):
                continue

            self.samples.append((pretrain_files[frame_idx], label_file))

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
        return [b for b in boxes if b is not None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pretrain_file, label_file = self.samples[idx]

        data   = torch.load(pretrain_file, weights_only=False)
        labels = self.load_labels(label_file)  # list of (8,3) arrays, may be empty

        cam = data["cam_bev"].float()

        if self.augment:
            cam, data["radar"], _ = self.augment_sample(cam, data["radar"]["points"])

        if self.img_mean is not None and self.img_std is not None:
            cam = (cam - self.img_mean) / (self.img_std + 1e-6)

        return {
            "cam_bev": cam,
            "radar":   data["radar"],
            "boxes":   labels,  # list of (8,3) np arrays, empty if no people in frame
        }

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