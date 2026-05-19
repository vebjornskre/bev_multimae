import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterPointHead(nn.Module):
    """
    CenterPoint detection head for BEV object detection.
    Takes spatial BEV features and outputs detection heatmaps and regressions.

    Architecture:
    - Input: (B, 128, 128, 128) spatial features from TokenToSpatialAdapter
    - Shared backbone: Series of Conv2d layers
    - Task heads:
      - Heatmap: 1 channel (object presence)
      - Regression: 2 channels (xy offset)
      - Height: 1 channel (z coordinate)
      - Dimensions: 3 channels (length, width, height)
      - Rotation: 2 channels (sin, cos of yaw)
    """

    def __init__(self, in_channels=128, num_tasks=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_tasks = num_tasks

        # Shared backbone: reduce spatial resolution and learn features
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # After backbone: (B, 128, 64, 64)

        # Task-specific heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        self.height_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.dim_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        self.rot_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, spatial_features):
        """
        Args:
            spatial_features: (B, 128, 128, 128) BEV spatial features

        Returns:
            detections dict with keys:
            - heatmap: (B, 1, 64, 64) object heatmap
            - reg: (B, 2, 64, 64) xy regression
            - height: (B, 1, 64, 64) z coordinate
            - dim: (B, 3, 64, 64) lwh dimensions
            - rot: (B, 2, 64, 64) sin/cos rotation
        """
        # Shared backbone
        feat = self.backbone(spatial_features)  # (B, 128, 64, 64)

        # Task heads
        heatmap = torch.sigmoid(self.heatmap_head(feat))
        reg = self.reg_head(feat)
        height = self.height_head(feat)
        dim = F.relu(self.dim_head(feat))
        rot = self.rot_head(feat)

        return {
            'heatmap': heatmap,
            'reg': reg,
            'height': height,
            'dim': dim,
            'rot': rot,
        }


class CenterPointDetector(nn.Module):
    """
    End-to-end CenterPoint detector combining spatial adapter and detection head.
    """

    def __init__(self, token_adapter, detection_head):
        super().__init__()
        self.token_adapter = token_adapter
        self.detection_head = detection_head

    def forward(self, encoder_tokens):
        """
        Args:
            encoder_tokens: (B, 649, 384) encoder output tokens

        Returns:
            detections dict
        """
        spatial_features = self.token_adapter(encoder_tokens)
        detections = self.detection_head(spatial_features)
        return detections
