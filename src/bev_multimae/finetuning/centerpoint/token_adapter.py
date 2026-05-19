import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenToSpatialAdapter(nn.Module):
    """
    Converts token sequence (B, 649, 384) back to spatial BEV (B, C, 128, 128).

    Token structure:
    - 1 global token
    - 324 radar tokens (18x18)
    - 324 camera tokens (18x18)

    Process:
    1. Extract and reshape tokens to spatial (B, 384, 18, 18)
    2. Optionally add global token via residual (if include_global=True)
    3. Concatenate radar and camera: (B, 768, 18, 18) or (B, 1152, 18, 18)
    4. Project down to (B, 128, 18, 18)
    5. Upsample to (B, 128, 128, 128)
    """

    def __init__(self, dim_tokens=384, output_channels=128, include_global=True):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.output_channels = output_channels
        self.grid_size = 18
        self.target_size = 128
        self.include_global = include_global

        # 1x1 convolution to project fused features
        in_channels = dim_tokens * 2 if not include_global else dim_tokens * 2
        self.fusion_proj = nn.Conv2d(
            in_channels,
            output_channels,
            kernel_size=1
        )

    def forward(self, encoder_tokens, task_masks=None):
        """
        Args:
            encoder_tokens: (B, 649, 384) - global + radar + camera tokens
            task_masks: dict with keys "radar" and "cam_bev", optional
                       used to reshape tokens correctly

        Returns:
            spatial_features: (B, output_channels, 128, 128)
        """
        batch_size = encoder_tokens.shape[0]

        # Extract tokens: global(1) + radar(324) + camera(324)
        global_tokens = encoder_tokens[:, 0:1, :]  # (B, 1, 384)
        radar_tokens = encoder_tokens[:, 1:325, :]  # (B, 324, 384)
        cam_tokens = encoder_tokens[:, 325:649, :]  # (B, 324, 384)

        # Reshape to spatial (B, 384, 18, 18)
        radar_spatial = radar_tokens.reshape(batch_size, self.grid_size, self.grid_size, self.dim_tokens)
        radar_spatial = radar_spatial.permute(0, 3, 1, 2)  # (B, 384, 18, 18)

        cam_spatial = cam_tokens.reshape(batch_size, self.grid_size, self.grid_size, self.dim_tokens)
        cam_spatial = cam_spatial.permute(0, 3, 1, 2)  # (B, 384, 18, 18)

        # Concatenate radar and camera
        fused = torch.cat([radar_spatial, cam_spatial], dim=1)  # (B, 768, 18, 18)

        # Optionally add global token via residual (cheap broadcast add)
        if self.include_global:
            global_spatial = global_tokens.reshape(batch_size, self.dim_tokens, 1, 1)  # (B, 384, 1, 1)
            fused[:, :self.dim_tokens] = fused[:, :self.dim_tokens] + global_spatial

        # Project down: (B, 768, 18, 18) -> (B, 128, 18, 18)
        fused = self.fusion_proj(fused)

        # Upsample: (B, 128, 18, 18) -> (B, 128, 128, 128)
        spatial_features = F.interpolate(
            fused,
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        )

        return spatial_features
