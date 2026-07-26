import torch
import torch.nn as nn


def _task_head(out_ch, dropout, mid_ch=64):
    return nn.Sequential(
        nn.Conv2d(128, mid_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.Conv2d(mid_ch, out_ch, kernel_size=1),
    )


def _backbone_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CenterPointHead(nn.Module):
    def __init__(self, in_channels=128, num_backbone_layers=2, dropout=0.0):
        super().__init__()

        if num_backbone_layers == 2:
            self.backbone = nn.Sequential(
                _backbone_block(in_channels, 128, stride=2),
                _backbone_block(128, 128),
            )
        elif num_backbone_layers == 3:
            self.backbone = nn.Sequential(
                _backbone_block(in_channels, 128),
                _backbone_block(128, 128),
                _backbone_block(128, 128, stride=2),
            )
        else:
            raise ValueError(f"num_backbone_layers must be 2 or 3, got {num_backbone_layers}")

        self.heatmap_head = _task_head(1, dropout, mid_ch=64)
        self.reg_head     = _task_head(2, dropout, mid_ch=64)
        self.height_head  = _task_head(1, dropout, mid_ch=64)
        self.dim_head     = _task_head(3, dropout, mid_ch=128)
        self.rot_head     = _task_head(2, dropout, mid_ch=128)

        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, spatial_features):
        feat = self.backbone(spatial_features)
        return {
            "heatmap": self.heatmap_head(feat),
            "reg":     self.reg_head(feat),
            "height":  self.height_head(feat),
            "dim":     torch.nn.functional.softplus(self.dim_head(feat)), # positive
            "rot":     self.rot_head(feat),
        }


class CenterPointDetector(nn.Module):
    def __init__(self, token_adapter, detection_head):
        super().__init__()
        self.token_adapter = token_adapter
        self.detection_head = detection_head

    def forward(self, encoder_tokens):
        return self.detection_head(self.token_adapter(encoder_tokens))