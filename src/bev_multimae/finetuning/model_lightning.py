import pytorch_lightning as pl
import torch
from bev_multimae.finetuning.centerpoint.losses import CenterPointLoss
from bev_multimae.finetuning.centerpoint.targets import build_centerpoint_targets_with_gaussian_gpu


class CenterPointLightning(pl.LightningModule):
    def __init__(
        self,
        encoder,
        detector,
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        num_encoded_tokens=288,
        heatmap_weight=1.0,
        offset_weight=1.0,
        height_weight=0.1,
        dim_weight=1.0,
        rot_weight=1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.detector = detector
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_encoded_tokens = num_encoded_tokens

        self.loss_fn = CenterPointLoss(
            heatmap_weight=heatmap_weight,
            offset_weight=offset_weight,
            height_weight=height_weight,
            dim_weight=dim_weight,
            rot_weight=rot_weight,
        )

        self.save_hyperparameters(ignore=["encoder", "detector"])

    def forward(self, batch):
        encoder_tokens, task_masks = self.encoder(
            batch,
            mask_inputs=False,
            num_encoded_tokens=self.num_encoded_tokens,
        )
        detections = self.detector(encoder_tokens)
        return detections

    def training_step(self, batch, batch_idx):
        detections = self(batch)

        # Build targets on GPU
        boxes_list = batch["boxes"]
        device = batch["cam_bev"].device
        targets = self._build_targets_batch(boxes_list, device)

        loss_dict = self.loss_fn(detections, targets)
        total_loss = loss_dict["total_loss"]

        batch_size = batch["cam_bev"].size(0)
        self.log("train/total_loss", total_loss, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        detections = self(batch)

        # Build targets on GPU
        boxes_list = batch["boxes"]
        device = batch["cam_bev"].device
        targets = self._build_targets_batch(boxes_list, device)

        loss_dict = self.loss_fn(detections, targets)
        total_loss = loss_dict["total_loss"]

        batch_size = batch["cam_bev"].size(0)
        self.log("val/total_loss", total_loss, batch_size=batch_size)

        return total_loss

    def _build_targets_batch(self, boxes_list, device):
        """Build detection targets on GPU for a batch."""
        detection_targets_list = []
        for boxes in boxes_list:
            # Build targets on GPU
            targets = build_centerpoint_targets_with_gaussian_gpu(
                boxes,
                bev_range=(-20, -20, 20, 20),
                grid_size=64,
                gaussian_radius=2,
                device=device
            )
            # Transpose to (C, H, W) format for batch stacking
            targets_transposed = {}
            for key in targets:
                if targets[key].dim() == 3:
                    targets_transposed[key] = targets[key].permute(2, 0, 1).unsqueeze(0)
                else:
                    targets_transposed[key] = targets[key].unsqueeze(0)
            detection_targets_list.append(targets_transposed)

        # Stack detection targets
        detection_targets = {}
        for key in detection_targets_list[0].keys():
            detection_targets[key] = torch.cat(
                [t[key] for t in detection_targets_list], dim=0
            )
        return detection_targets

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
