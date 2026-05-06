import torch
from torch.utils.data import DataLoader
from einops import rearrange
from torch.utils.data import ConcatDataset

from bev_multimae.datasets.data import BEVDataset, collate_radar


def denorm_patches(pred, target, patch_size):
    p = patch_size
    H, W = target.shape[-2:]
    nh, nw = H // p, W // p

    t = rearrange(target, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
    mean = t.mean(dim=-1, keepdim=True)
    var = t.var(dim=-1, keepdim=True)

    p_pred = rearrange(pred, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
    p_pred = p_pred * torch.sqrt(var + 1e-6) + mean

    return rearrange(p_pred, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)",
                     nh=nh, nw=nw, p1=p, p2=p, c=target.shape[1])


def denorm_img(img, mean, std):
    return img * (std + 1e-6) + mean

def compute_img_stats(data_paths):
    ds = ConcatDataset([BEVDataset(p, split="train") for p in data_paths])
    loader = DataLoader(ds, batch_size=8, num_workers=4, collate_fn=collate_radar)

    mean = 0.
    std = 0.
    n = 0

    for batch in loader:
        x = batch["cam_bev"]
        b = x.size(0)
        x = x.view(b, x.size(1), -1)
        mean += x.mean(2).sum(0)
        std  += x.std(2).sum(0)
        n += b

    mean /= n
    std  /= n
    return mean.clone().detach(), std.clone().detach()


def compute_radar_stats(data_paths):
    ds = ConcatDataset([BEVDataset(p, split="train") for p in data_paths])
    loader = DataLoader(ds, batch_size=8, num_workers=4, collate_fn=collate_radar)

    mean = 0.
    std = 0.
    n = 0

    for batch in loader:
        x = batch["radar"]["points"][:, 4:7].float()
        mean += x.mean(0)
        std  += x.std(0)
        n += 1

    mean /= n
    std  /= n
    return mean, std