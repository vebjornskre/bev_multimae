import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from bev_multimae.multimae.criterion import MaskedMSELoss, MaskedL1Loss

class BevMultiMAELightning(pl.LightningModule):
    def __init__(self, model, lr=1e-4, weight_decay=0.01, num_encoded_tokens=288, 
             norm_pix=False, depth=6, num_heads=8, dim_tokens=256, warmup_steps=500, 
             drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0, data_aug=False, num_rad_channels=11):
        super().__init__()
        self.model = model
        self.lr = lr
        self.cam_loss = MaskedMSELoss(patch_size=15, stride=1, norm_pix=norm_pix)
        self.cam_l1   = MaskedL1Loss(patch_size=15, stride=1, norm_pix=norm_pix)
        self.rad_loss = MaskedL1Loss(patch_size=1, stride=1, norm_pix=False)
        self.num_encoded_tokens = num_encoded_tokens
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        self.save_hyperparameters(ignore=["model"])
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds, task_masks = self.model(batch, mask_inputs=True, num_encoded_tokens=self.num_encoded_tokens)
        loss = self.compute_loss(preds, batch, task_masks)
        batch_size = batch["cam_bev"].size(0)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, task_masks = self.model(batch, mask_inputs=True, num_encoded_tokens=self.num_encoded_tokens)
        loss = self.compute_loss(preds, batch, task_masks)
        batch_size = batch["cam_bev"].size(0)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        return loss

    def gradient_loss(self, pred, target, mask, patch_size=15):
        dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])
        dy = (pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])

        H, W = pred.shape[-2:]
        nh, nw = H // patch_size, W // patch_size

        # expand mask to pixel space
        m = mask.reshape(pred.shape[0], nh, nw)
        m = m.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2).unsqueeze(1).float()

        dx_loss = (dx.abs() * m[:, :, :, 1:]).sum() / m[:, :, :, 1:].sum().clamp(min=1)
        dy_loss = (dy.abs() * m[:, :, 1:]).sum() / m[:, :, 1:].sum().clamp(min=1)

        return dx_loss + dy_loss


    def compute_loss(self, preds, batch, task_masks):
        cam = self.cam_loss(preds["cam_bev"], batch["cam_bev"], task_masks["cam_bev"])
        l1  = self.cam_l1(preds["cam_bev"], batch["cam_bev"], task_masks["cam_bev"])
        # grad = self.gradient_loss(preds["cam_bev"], batch["cam_bev"], task_masks["cam_bev"])
        camera_weight = 5

        rad_pred   = preds["radar"]
        rad_target = batch["radar_target"].to(rad_pred.device)

        occ_loss = F.binary_cross_entropy_with_logits(
            rad_pred[:, 0:1], rad_target[:, 0:1],
            pos_weight=torch.tensor(10.0).to(rad_pred.device)
        )

        occ_mask  = (rad_target[:, 0:1] > 0.5).float()
        reg_loss  = self.rad_loss(rad_pred[:, 1:] * occ_mask, rad_target[:, 1:] * occ_mask, task_masks["radar"])

        cam_loss  = camera_weight * (cam + 0.1 * l1)
        rad_loss  = occ_loss + reg_loss


        return cam_loss + rad_loss

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "bias" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - self.warmup_steps, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }