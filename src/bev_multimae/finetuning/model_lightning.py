import pytorch_lightning as pl
import torch
from bev_multimae.finetuning.centerpoint.losses import CenterPointLoss


class CenterPointLightning(pl.LightningModule):
    def __init__(
        self,
        encoder,
        detector,
        lr=1e-4,
        encoder_lr=1e-6,
        weight_decay=0.01,
        warmup_steps=500,
        heatmap_weight=1.0,
        offset_weight=1.0,
        height_weight=0.1,
        dim_weight=1.0,
        rot_weight=1.0,
        modality_dropout=False,
        drop_radar_prob=0.0,
        drop_cam_prob=0.0,
        freeze_encoder=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.detector = detector
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.modality_dropout = modality_dropout
        self.drop_radar_prob = drop_radar_prob
        self.drop_cam_prob = drop_cam_prob
        self.freeze_encoder = freeze_encoder

        self.loss_fn = CenterPointLoss(
            heatmap_weight=heatmap_weight,
            offset_weight=offset_weight,
            height_weight=height_weight,
            dim_weight=dim_weight,
            rot_weight=rot_weight,
        )

        self.save_hyperparameters(ignore=["encoder", "detector"])

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder and mode:
            self.encoder.eval()
        return self

    def forward(self, batch):
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_tokens, _ = self.encoder(batch, mask_inputs=False)
            encoder_tokens = encoder_tokens.detach()
        else:
            encoder_tokens, _ = self.encoder(batch, mask_inputs=False)
        return self.detector(encoder_tokens)

    def _log_losses(self, prefix, loss_dict, batch_size):
        for key, val in loss_dict.items():
            self.log(f"{prefix}/{key}", val, batch_size=batch_size, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        batch = self.apply_modality_dropout(batch)
        loss_dict = self.loss_fn(self(batch), batch["targets"])
        self._log_losses("train", loss_dict, batch["cam_bev"].size(0))
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.loss_fn(self(batch), batch["targets"])
        batch_size = batch["cam_bev"].size(0)
        self._log_losses("val", loss_dict, batch_size)
        self.log("val_total_loss", loss_dict["total_loss"], batch_size=batch_size, on_step=False, on_epoch=True)
        return loss_dict["total_loss"]

    def configure_optimizers(self):
        def group_params(module, lr):
            decay, no_decay = [], []
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or "bias" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)

            groups = []
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": self.weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
            return groups

        param_groups = (
            group_params(self.detector, self.lr)
            + group_params(self.encoder, self.encoder_lr)
        )

        optimizer = torch.optim.AdamW(param_groups)

        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup_steps = min(self.warmup_steps, max(0, total_steps - 1))

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_steps - warmup_steps),
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def apply_modality_dropout(self, batch):
        if not self.training or not self.modality_dropout:
            return batch

        r = torch.rand(1, device=self.device).item()
        batch = dict(batch)

        if r < self.drop_radar_prob:
            radar = dict(batch["radar"])
            radar["points"] = radar["points"].clone()
            radar["points"][:, 1:] = 0

            if "f_cluster" in radar:
                radar["f_cluster"] = torch.zeros_like(radar["f_cluster"])

            if "f_center" in radar:
                radar["f_center"] = torch.zeros_like(radar["f_center"])

            batch["radar"] = radar

        elif r < self.drop_radar_prob + self.drop_cam_prob:
            batch["cam_bev"] = batch["cam_bev"].clone() * 0

        return batch