from pathlib import Path
from torch.utils.data import Dataset
import torch
import os

def collate_radar(samples):

    cam_bevs = torch.stack([s["cam_bev"] for s in samples])
    radar_targets = torch.stack([s["radar_target"] for s in samples])
    
    radar_list = [s["radar"] for s in samples]
    radar_batch = {}
    
    for k in radar_list[0].keys():
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
    def __init__(self, data_path, split="train", img_mean=None, img_std=None, rad_mean=None, rad_std=None):
        assert split in ['train', 'val', 'test']
        self.meta = torch.load(os.path.join(data_path, 'meta.pt'), weights_only=False)
        skip = {"meta.pt", "radar_stats.pt"}

        skip_events = {
            "evt_0dpi1jtdkfL4ReZx", "evt_0e3qdKh444ogAO7k",
            "evt_0e3qWOZPmBcxw6le", "evt_0e3qWOZPmBcxw6le",
            "evt_0e3qYFzAbTyiJEC0", "evt_0e3qZ3gUUyKSySmG",
            "evt_0e8RBM8c9lbnJIcY", "evt_0e8RGcuJoVLx95HV",
            "evt_0e8RGh9h8HVd0mMN", "evt_0e8RHpfQW2EAsyh5",
            "evt_0e8RKuQ17gJsASht", "evt_0e8RPpu7MFbAyMMZ",
            "evt_0e8RPtAGjsFkH7iI", "evt_0e8RSXbBo3Nf3CKl",
            "evt_0e8RPwdTi1LVwWTv", "evt_0e8RQ2Rhx24bzKAg",
            "evt_0e8RQLQbupv7ze9a", "evt_0e8RQQaclY2VF8AH",
            "evt_0e3qWOZPmBcxw6le"
            
            }  # folders to skip

        self.files = sorted(
            [
                p for p in Path(os.path.join(data_path, split)).rglob("*.pt")
                if p.name not in skip
                and not any(parent.name in skip_events for parent in p.parents)
            ],
            key=lambda p: int(p.stem.split("_")[-1])
        )

        # Overfitted model uses the image below:
        # self.files = [os.path.join(data_path,'train','001650.pt')]

        self.img_mean = img_mean.view(3,1,1) if img_mean is not None else None
        self.img_std  = img_std.view(3,1,1) if img_std is not None else None

        self.rad_mean = (rad_mean.view(3) if rad_mean is not None else None)
        self.rad_std  = (rad_std.view(3) if rad_std  is not None else None)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)

        if self.img_mean is not None and self.img_std is not None:
            data["radar"]["points"][:, 4:7] = (data["radar"]["points"][:, 4:7] - self.rad_mean) / (self.rad_std + 1e-6)

        cam = data["cam_bev"].float()
        if self.img_mean is not None and self.img_std is not None:
            cam = (cam - self.img_mean) / (self.img_std + 1e-6)

        return {
            "cam_bev": cam,
            "radar": data["radar"],
            "radar_target": data["radar_target"].squeeze(0),
        }
    