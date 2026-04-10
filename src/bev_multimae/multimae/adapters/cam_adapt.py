# Adapter from camera to encoder
# Adapter from radar to ecoder
import torch
from torch import Tensor
import math
import torch.nn as nn
from bev_multimae.preprocessing.BEV.dynamic_pillar_vfe import PointPillarScatter
from bev_multimae.multimae.model_utils import positional_encoding_2d
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class CameraAdapter(nn.Module):
    def __init__(self, d_model, channels, patch_size, grid_size_hires):
        super().__init__()

        self.grid_size_hires = grid_size_hires
        self.patch_h, self.patch_w = patch_size
        self.d_model = d_model
        self.task_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.task_embedding, std=0.02)

        H, W = self.grid_size_hires[:2]
        patch_dim = channels * self.patch_h * self.patch_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_h, p2=self.patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.positional_embedding = positional_encoding_2d(
            nph = H // self.patch_h, 
            npw = W // self.patch_w,
            dim = d_model,
        )
     
    def forward(self, img):
        tokens = self.to_patch_embedding(img)
        tokens = tokens + self.positional_embedding.to(tokens.device).unsqueeze(0)
        tokens = tokens + self.task_embedding
        return tokens