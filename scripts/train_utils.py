import torch
from torch.utils.data import DataLoader
from bev_multimae.datasets.data import BEVDataset, collate_radar


def get_data_mean_std(train_dataset):

    loader = DataLoader(train_dataset, batch_size=8, num_workers=4, collate_fn=collate_radar)

    mean = 0.
    std = 0.
    n = 0

    for batch in loader:
        x = batch["cam_bev"]  # [B, C, H, W]
        b = x.size(0)
        
        x = x.view(b, x.size(1), -1)  # [B, C, HW]
        
        mean += x.mean(2).sum(0)
        std  += x.std(2).sum(0)
        n += b

    mean /= n
    std /= n

    print("mean:", mean)
    print("std:", std)
    return mean, std