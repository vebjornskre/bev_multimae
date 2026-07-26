import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from bev_multimae.multimae.criterion import MaskedMSELoss, MaskedL1Loss

class BevMultiMAELightning(pl.LightningModule):
    def __init__(self, model, lr=1e-4, weight_decay=0.01, num_encoded_tokens=288, 
         norm_pix=False, depth=6, num_heads=8, dim_tokens=256, warmup_steps=500, 
         drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0, data_aug=False,
         num_rad_channels=11, feat_patch_size=3, feat_weight=1.0, camera_weight=5.0, rad_weight=1.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.cam_loss = MaskedMSELoss(patch_size=15, stride=1, norm_pix=norm_pix)
        self.cam_l1   = MaskedL1Loss(patch_size=15, stride=1, norm_pix=norm_pix)
        self.rad_loss_l1 = MaskedL1Loss(patch_size=1, stride=1, norm_pix=False)
        self.rad_loss_mse = MaskedMSELoss(patch_size=1, stride=1, norm_pix=False)

        self.feat_loss_mse = MaskedMSELoss(patch_size=feat_patch_size, stride=1, norm_pix=False)
        self.feat_patch_size = feat_patch_size

        self.feat_weight = feat_weight
        self.camera_weight = camera_weight
        self.rad_weight = rad_weight

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
        loss_dict = self.compute_loss(preds, batch, task_masks)
        batch_size = batch["cam_bev"].size(0)

        self.log("train_loss", loss_dict["total_loss"], prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("train/cam_loss", loss_dict["cam_loss"]/self.camera_weight, on_epoch=True, batch_size=batch_size)
        self.log("train/rad_loss", loss_dict["rad_loss"]/self.rad_weight, on_epoch=True, batch_size=batch_size)
        self.log("train/feat_loss", loss_dict["feat_loss"]/self.feat_weight, on_epoch=True, batch_size=batch_size)

        return loss_dict["total_loss"]


    def validation_step(self, batch, batch_idx):
        preds, task_masks = self.model(batch, mask_inputs=True, num_encoded_tokens=self.num_encoded_tokens)

        loss_dict = self.compute_loss(preds, batch, task_masks)
        batch_size = batch["cam_bev"].size(0)

        cam_loss = loss_dict["cam_loss"] / self.camera_weight
        rad_loss = loss_dict["rad_loss"] / self.rad_weight
        feat_loss = loss_dict["feat_loss"] / self.feat_weight
        val_loss_unweighted = cam_loss + rad_loss + feat_loss

        self.log("val_loss", loss_dict["total_loss"], prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_total_loss_unweighted", val_loss_unweighted, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val/cam_loss", cam_loss, on_epoch=True, batch_size=batch_size)
        self.log("val/rad_loss", rad_loss, on_epoch=True, batch_size=batch_size)
        self.log("val/feat_loss", feat_loss, on_epoch=True, batch_size=batch_size)

        return loss_dict["total_loss"]


    def compute_loss(self, preds, batch, task_masks):
        cam = self.cam_loss(preds["cam_bev"], batch["cam_bev"], task_masks["cam_bev"])
        l1 = self.cam_l1(preds["cam_bev"], batch["cam_bev"], task_masks["cam_bev"])

        rad_pred = preds["radar"]
        rad_target = batch["radar_target"].to(rad_pred.device)

        occ_loss = F.binary_cross_entropy_with_logits(
            rad_pred[:, 0:1],
            rad_target[:, 0:1],
            pos_weight=torch.tensor(10.0, device=rad_pred.device),
        )

        occ_mask = (rad_target[:, 0:1] > 0.5).float()

        reg_loss_1 = self.rad_loss_l1(
            rad_pred[:, 1:9] * occ_mask,
            rad_target[:, 1:9] * occ_mask,
            task_masks["radar"],
        )

        reg_loss_2 = self.rad_loss_mse(
            rad_pred[:, 9:11] * occ_mask,
            rad_target[:, 9:11] * occ_mask,
            task_masks["radar"],
        )

        feat_target = batch["bev_feat"].to(preds["bev_feat"].device).float()

        raw_feat_loss = self.feat_loss_mse(
            preds["bev_feat"],
            feat_target,
            task_masks["bev_feat"],
        )

        cam_loss = self.camera_weight * (cam + 0.1 * l1)
        rad_loss = self.rad_weight * (occ_loss + reg_loss_1 + reg_loss_2)
        feat_loss = self.feat_weight * raw_feat_loss

        total_loss = cam_loss + rad_loss + feat_loss

        return {
            "total_loss": total_loss,
            "cam_loss": cam_loss.detach(),
            "rad_loss": rad_loss.detach(),
            "feat_loss": feat_loss.detach(),
        }

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