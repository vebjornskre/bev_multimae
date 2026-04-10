from pathlib import Path
from torch.utils.data import Dataset
import torch
import os

def collate_radar(samples):
    cam_bevs = torch.stack([s["cam_bev"] for s in samples])
    
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
    
    radar_batch["points"] = torch.cat(points_list)
    radar_batch["pillar_coords"] = torch.cat(coords_list)
    
    return {"cam_bev": cam_bevs, "radar": radar_batch}

class BEVDataset(Dataset):
    def __init__(self, data_path, cfg):
        self.meta = torch.load(os.path.join(cfg.processed_data_dir, 'meta.pt'), weights_only=False)
        skip = {"meta.pt", "radar_stats.pt"}
        self.files = sorted(
            [p for p in Path(data_path).rglob("*.pt") if p.name not in skip],
            key=lambda p: int(p.stem.split("_")[-1])
        )
        self.mean = self.meta["radar_mean"]
        self.std = self.meta["radar_std"]
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        data["radar"]["points"][:, 4:7] = (data["radar"]["points"][:, 4:7] - self.mean) / (self.std + 1e-6)
        
        cam = data["cam_bev"].float() / 255.0
        cam = (cam - self.imagenet_mean) / self.imagenet_std
        
        return {
            "cam_bev": cam,
            "radar": data["radar"],
        }
    