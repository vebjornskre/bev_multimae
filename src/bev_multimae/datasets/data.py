from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import random
from bev_multimae.preprocessing.BEV.dynamic_pillar_vfe import DynamicPillarizer, build_bev_target
import torchvision
import math

def collate_radar(samples):

    cam_bevs = torch.stack([s["cam_bev"] for s in samples])
    radar_targets = torch.stack([s["radar_target"] for s in samples])
    
    radar_list = [s["radar"] for s in samples]
    radar_batch = {}
    
    for k in radar_list[0].keys():
        if k == "batch_size":
            continue
        radar_batch[k] = torch.cat([r[k] for r in radar_list], dim=0)
    
    # remap batch indices so sample 0 has idx 0, sample 1 has idx 1, etc.
    offset = 0
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
    for i, r in enumerate(radar_list):
        inv = r["pillar_inv"].clone() + pillar_offset
        inv_list.append(inv)
        pillar_offset += r["pillar_coords"].shape[0]

    radar_batch["pillar_inv"] = torch.cat(inv_list)
    
    radar_batch["points"] = torch.cat(points_list)
    radar_batch["pillar_coords"] = torch.cat(coords_list)
    radar_batch["batch_size"] = len(samples)

    return {
        "cam_bev": cam_bevs, 
        "radar": radar_batch,
        "radar_target": radar_targets,  # and this
    }

class BEVDataset(Dataset):
    def __init__(
        self, data_path, split="train", 
        img_mean=None, img_std=None, rad_mean=None, 
        rad_std=None, augment=False,
        v_flip_rate=0.0, h_flip_rate=0.0, rot_rate=0.0, rot_angle=(-20, 20),
        point_cloud_range=None, num_rad_channels=11
        ):
        assert split in ['train', 'val', 'test']
        self.meta = torch.load(os.path.join(data_path, 'meta.pt'), weights_only=False)
        skip = {"meta.pt", "radar_stats.pt"}

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
            
            }  # folders to skip

        self.files = sorted(
            [
                p for p in Path(os.path.join(data_path, split)).rglob("*.pt")
                if p.name not in skip
                and not any(parent.name in skip_events for parent in p.parents)
            ],
            key=lambda p: int(p.stem.split("_")[-1])
        )

        self.img_mean = img_mean.view(3,1,1) if img_mean is not None else None
        self.img_std  = img_std.view(3,1,1) if img_std is not None else None

        self.rad_mean = (rad_mean.view(3) if rad_mean is not None else None)
        self.rad_std  = (rad_std.view(3) if rad_std  is not None else None)

        self.grid_size = self.meta['grid_size']
        self.num_rad_channels = num_rad_channels

        self.augment = augment
        if self.augment:
            self.pillarizer = DynamicPillarizer(
                voxel_size=self.meta['voxel_size'],
                grid_size=self.meta['grid_size'],
                point_cloud_range=point_cloud_range
            )
            self.rot_rate = rot_rate
            self.h_flip_rate = h_flip_rate
            self.v_flip_rate = v_flip_rate
            self.rot_angle = rot_angle
            self.num_rad_channels = num_rad_channels
        

    def augment_sample(self, cam, rad_points):
        pcr = self.pillarizer.point_cloud_range  # [x_min, y_min, z_min, x_max, y_max, z_max]
        x_center = (pcr[0] + pcr[3]) / 2
        y_center = (pcr[1] + pcr[4]) / 2

        if self.h_flip_rate and random.random() < self.h_flip_rate:
            cam = torch.flip(cam, dims=[-2])
            rad_points[:, 2] = 2 * y_center - rad_points[:, 2]  # mirror around y center

        if self.v_flip_rate and random.random() < self.v_flip_rate:
            cam = torch.flip(cam, dims=[-1])
            rad_points[:, 1] = 2 * x_center - rad_points[:, 1]  # mirror around x center

        if self.rot_rate and random.random() < self.rot_rate:
            angle_deg = random.uniform(self.rot_angle[0], self.rot_angle[1])

            angle_rad = math.radians(angle_deg)
            cos, sin = math.cos(angle_rad), math.sin(angle_rad)
            # rotate around PCR center
            xy = rad_points[:, 1:3].clone()
            cx, cy = x_center.item(), y_center.item()
            rad_points[:, 1] = cx + cos * (xy[:, 0] - cx) - sin * (xy[:, 1] - cy)
            rad_points[:, 2] = cy + sin * (xy[:, 0] - cx) + cos * (xy[:, 1] - cy)
            cam = torchvision.transforms.functional.rotate(cam, -angle_deg)

        batch_dict = self.pillarizer.forward(rad_points)
        batch_dict["batch_size"] = 1
        bev_target = build_bev_target(batch_dict, grid_size=self.grid_size[:2], num_rad_channels=self.num_rad_channels)

        return cam, batch_dict, bev_target


    def __len__(self):  
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)

        data["radar_target"] = build_bev_target(
            data["radar"], 
            grid_size=self.grid_size[:2], 
            num_rad_channels=self.num_rad_channels
        )

        cam = data["cam_bev"].float()

        if self.augment:
            cam, data["radar"], data["radar_target"] = self.augment_sample(cam, data["radar"]["points"])

        if self.rad_mean is not None and self.rad_std is not None:
            data["radar"]["points"][:, 4:7] = (data["radar"]["points"][:, 4:7] - self.rad_mean) / (self.rad_std + 1e-6)

        if self.img_mean is not None and self.img_std is not None:
            cam = (cam - self.img_mean) / (self.img_std + 1e-6)

        return {
            "cam_bev": cam,
            "radar": data["radar"],
            "radar_target": data["radar_target"].squeeze(0),
        }
    