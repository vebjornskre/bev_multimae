# Adapter from radar to ecoder
import torch
from torch import Tensor
import math
import torch.nn as nn
from bev_multimae.preprocessing.BEV.dynamic_pillar_vfe import PointPillarScatter
from bev_multimae.multimae.model_utils import positional_encoding_2d
import matplotlib.pyplot as plt


class RadarAdapter(nn.Module):
    def __init__(self, d_model, grid_size, num_point_features, num_vfe_features):
        super().__init__()

        self.grid_size = grid_size
        self.num_point_features = num_point_features
        self.num_vfe_features = num_vfe_features
        self.d_model = d_model
        self.task_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.task_embedding, std=0.02)

        H, W = self.grid_size[:2]

        self.scatter = PointPillarScatter(
            grid_size=grid_size[:2],
            num_point_features=num_point_features,
            num_filters=num_vfe_features
        )

        self.positional_embedding = positional_encoding_2d(
            nph = H,
            npw = W,
            dim = d_model,
        )
        self.proj = nn.Conv2d(self.scatter.out_features, d_model, 1)
    
    def forward(self, batch_dict):
        batch_dict = self.scatter(batch_dict)
        x = batch_dict["spatial_features"]        # [B, C, H, W]
        x = self.proj(x)                          # [B, d_model, H, W]
        x = x.flatten(2).transpose(1, 2)          # [B, N, d_model]
        pos = self.positional_embedding.to(x.device)  # [N, d_model]
        x = x + pos.unsqueeze(0)                  # [B, N, d_model]
        x = x + self.task_embedding
        return x