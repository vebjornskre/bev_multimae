import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from bev_multimae.multimae.criterion import MaskedMSELoss

class BevMultiMAELightning(pl.LightningModule):
    def __init__(self, model, lr=1e-4, num_encoded_tokens=288):
        super().__init__()
        self.model = model
        self.lr = lr
        self.cam_loss = MaskedMSELoss(patch_size=15, stride=1, norm_pix=True)
        self.rad_loss = MaskedMSELoss(patch_size=1, stride=1, norm_pix=False)
        self.num_encoded_tokens = num_encoded_tokens
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds, task_masks = self.model(batch, mask_inputs=True, num_encoded_tokens=self.num_encoded_tokens)
        loss = self.compute_loss(preds, batch, task_masks)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def gradient_loss(self, pred, target):
        dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])
        dy = (pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])
        return dx.abs().mean() + dy.abs().mean()


    def compute_loss(self, preds, batch, task_masks):
        cam = self.cam_loss(preds["cam_bev"], batch["cam_bev"], task_masks["cam_bev"])
        grad = self.gradient_loss(preds["cam_bev"], batch["cam_bev"])
       
        rad_pred = preds["radar"]
        rad_target = batch["radar_target"].to(rad_pred.device)
        
        # channel 0: binary occupancy
        occ_loss = F.binary_cross_entropy_with_logits(
            rad_pred[:, 0:1], rad_target[:, 0:1],
            pos_weight=torch.tensor(10.0).to(rad_pred.device)
        )
        
        # # channels 1-8: continuous regression
        reg_loss = self.rad_loss(rad_pred[:, 1:], rad_target[:, 1:], task_masks["radar"])
        
        return cam + 1 * grad + occ_loss + reg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)