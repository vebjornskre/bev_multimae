import torch
import torch.nn as nn
import torch.nn.functional as F


class FastFocalLoss(nn.Module):
    """
    Focal loss for dense heatmap prediction.
    Adapted from CenterPoint to work with dense targets instead of sparse.

    Args:
        out: (B, 1, H, W) predicted heatmap
        target: (B, 1, H, W) target heatmap with Gaussian blobs
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target):
        """
        Args:
            out: (B, 1, H, W) predicted heatmap [0, 1]
            target: (B, 1, H, W) target heatmap [0, 1]
        """
        # Focal loss: down-weight easy negatives
        # For background (target=0): loss = log(1-out) * out^2
        # For foreground (target=1): loss = log(out) * (1-out)^2

        out = torch.clamp(out, min=1e-4, max=1.0 - 1e-4)

        # Positive (object present)
        pos_loss = -(1 - target) ** 4 * torch.log(1 - out)

        # Negative (background)
        neg_loss = -target * torch.log(out) * torch.pow(1 - out, 2)

        loss = pos_loss + neg_loss
        return loss.mean()


class RegLoss(nn.Module):
    """
    Regression loss for dense targets.
    Adapted from CenterPoint to work with dense targets instead of sparse.

    Handles: offset (reg), height, dimensions, rotation
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (B, C, H, W) predicted regression targets (offset, height, dim, rot)
            target: (B, C, H, W) target regression targets
            mask: (B, 1, H, W) binary mask indicating valid locations (object centers)
                  If None, uses target > 0 to infer valid locations
        """
        # Use L1 loss
        loss = F.l1_loss(pred, target, reduction='none')

        # Apply mask if provided, otherwise mask where target is non-zero
        if mask is None:
            # For multi-channel targets, create mask from any channel
            mask = (target.abs().sum(dim=1, keepdim=True) > 0).float()
        else:
            mask = mask.float()

        # Mask the loss (only supervise at object locations)
        masked_loss = loss * mask

        # Normalize by number of positive locations
        num_pos = mask.sum() + 1e-4
        normalized_loss = masked_loss.sum() / num_pos

        return normalized_loss


class CenterPointLoss(nn.Module):
    """
    Combined loss for CenterPoint detection with dense targets.
    """

    def __init__(self, heatmap_weight=1.0, offset_weight=1.0, height_weight=0.1,
                 dim_weight=1.0, rot_weight=1.0):
        super(CenterPointLoss, self).__init__()
        self.heatmap_loss_fn = FastFocalLoss()
        self.reg_loss_fn = RegLoss()

        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.height_weight = height_weight
        self.dim_weight = dim_weight
        self.rot_weight = rot_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys ['heatmap', 'reg', 'height', 'dim', 'rot']
                        each with shape (B, C, H, W)
            targets: dict with same keys, same shapes

        Returns:
            loss_dict: dict with individual losses and total loss
        """
        # Extract predictions and targets
        pred_heatmap = predictions['heatmap']
        pred_reg = predictions['reg']
        pred_height = predictions['height']
        pred_dim = predictions['dim']
        pred_rot = predictions['rot']

        target_heatmap = targets['heatmap']
        target_reg = targets['reg']
        target_height = targets['height']
        target_dim = targets['dim']
        target_rot = targets['rot']
        mask = targets['masks']  # (B, 1, H, W)

        # Compute losses
        heatmap_loss = self.heatmap_loss_fn(pred_heatmap, target_heatmap)

        # Regression losses (only at object centers)
        reg_loss = self.reg_loss_fn(pred_reg, target_reg, mask)
        height_loss = self.reg_loss_fn(pred_height, target_height, mask)
        dim_loss = self.reg_loss_fn(pred_dim, target_dim, mask)
        rot_loss = self.reg_loss_fn(pred_rot, target_rot, mask)

        # Weighted sum
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.offset_weight * reg_loss +
            self.height_weight * height_loss +
            self.dim_weight * dim_loss +
            self.rot_weight * rot_loss
        )

        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss.item(),
            'reg_loss': reg_loss.item(),
            'height_loss': height_loss.item(),
            'dim_loss': dim_loss.item(),
            'rot_loss': rot_loss.item(),
        }
