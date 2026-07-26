import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FiLM(nn.Module):
    def __init__(self, token_dim, feature_channels):
        super().__init__()
        self.proj = nn.Linear(token_dim, feature_channels * 2)

    def forward(self, x, global_tok):
        gamma, beta = self.proj(global_tok).chunk(2, dim=-1)
        gamma = gamma.reshape(gamma.shape[0], -1, 1, 1) + 1.0
        beta = beta.reshape(beta.shape[0], -1, 1, 1)
        return x * gamma + beta


class TokenToSpatialAdapter(nn.Module):
    def __init__(self, dim_tokens=384, output_channels=128, include_global=True):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.grid_size = 18
        self.include_global = include_global

        self.fusion_proj = nn.Conv2d(dim_tokens * 3, output_channels, kernel_size=1)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, 2, stride=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, 2, stride=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
            conv_bn_relu(output_channels, output_channels, kernel_size=3, padding=1),
        )

        if include_global:
            self.film1 = FiLM(dim_tokens, output_channels)
            self.film2 = FiLM(dim_tokens, output_channels)
            self.film3 = FiLM(dim_tokens, output_channels)

    def _to_spatial(self, tokens, B):
        g = self.grid_size
        return tokens.reshape(B, g, g, self.dim_tokens).permute(0, 3, 1, 2).contiguous()

    def forward(self, encoder_tokens, task_masks=None):
        B = encoder_tokens.shape[0]

        radar_spatial = self._to_spatial(encoder_tokens[:, 0:324, :], B)
        cam_spatial = self._to_spatial(encoder_tokens[:, 324:648, :], B)
        feat_spatial = self._to_spatial(encoder_tokens[:, 648:972, :], B)
        global_tok = encoder_tokens[:, -1, :]

        x = self.fusion_proj(torch.cat([radar_spatial, cam_spatial, feat_spatial], dim=1))

        x = self.film1(self.up1(x), global_tok) if self.include_global else self.up1(x)
        x = self.film2(self.up2(x), global_tok) if self.include_global else self.up2(x)
        x = self.film3(self.up3(x), global_tok) if self.include_global else self.up3(x)

        return x