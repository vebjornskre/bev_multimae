import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from bev_multimae.multimae.model_utils import positional_encoding_2d


class FeatureAdapter(nn.Module):
    def __init__(self, d_model, channels=384, patch_size=(3, 3), bev_feat_grid_size=(54, 54)):
        super().__init__()

        self.bev_feat_grid_size = tuple(bev_feat_grid_size)
        self.patch_h, self.patch_w = patch_size
        self.d_model = d_model
        self.channels = channels

        H, W = self.bev_feat_grid_size

        if H % self.patch_h != 0 or W % self.patch_w != 0:
            raise ValueError(
                f"bev_feat_grid_size {self.bev_feat_grid_size} must be divisible by patch_size {patch_size}"
            )

        patch_dim = channels * self.patch_h * self.patch_w

        self.task_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.task_embedding, std=0.02)

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_h,
                p2=self.patch_w,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.register_buffer(
            "positional_embedding",
            positional_encoding_2d(
                nph=H // self.patch_h,
                npw=W // self.patch_w,
                dim=d_model,
            ),
        )

    def forward(self, bev_feat):
        if bev_feat.ndim != 4:
            raise ValueError(f"Expected bev_feat shape (B, C, H, W), got {bev_feat.shape}")

        B, C, H, W = bev_feat.shape

        if C != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {C}")

        if (H, W) != self.bev_feat_grid_size:
            raise ValueError(
                f"Expected spatial size {self.bev_feat_grid_size}, got {(H, W)}"
            )

        tokens = self.to_patch_embedding(bev_feat)
        tokens = tokens + self.positional_embedding.unsqueeze(0).to(tokens.device)
        tokens = tokens + self.task_embedding

        return tokens