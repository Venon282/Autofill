import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model.vae.submodel.registry import *


class PlVAE(pl.LightningModule):
    def __init__(self, config):
        super(PlVAE, self).__init__()
        if config is None and not hasattr(self, 'config'):
            raise ValueError("Configuration dictionary is required for PlVAE.")
        if config is not None:
            self.config = config

        self.beta = config["training"]["beta"]

        model_class = self.config["model"]["vae_class"]
        self.model = MODEL_REGISTRY.get(model_class)(**self.config["model"]["args"])
        self.save_hyperparameters()

    def forward(self, x):
        y = x["data_y"]
        q = x["data_q"]
        metadata = x["metadata"]
        return self.model(y=y, q=q, metadata=metadata)

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def compute_loss(self, batch, output):
        x = batch["data_y"]
        recon = output['recon']
        mu = output['mu']
        logvar = output['logvar']

        recon_loss = F.mse_loss(recon, x, reduction='mean')
        # kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + self.beta * kl_div, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        y = batch["data_y"]
        q = batch["data_q"]
        metadata = batch["metadata"]

        output = self.model(y=y, q=q, metadata=metadata)
        loss, recon_loss, kl_loss = self.compute_loss(batch, output)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["data_y"]
        q = batch["data_q"]
        metadata = batch["metadata"]

        output = self.model(y=y, q=q, metadata=metadata)
        loss, recon_loss, kl_loss = self.compute_loss(batch, output)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["max_lr"])

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config["training"]["T_max"],
            eta_min=self.config["training"]["eta_min"])

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"}}

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['config'] = self.config
    #     return checkpoint
    #
    # def on_load_checkpoint(self, checkpoint):
    #     self.config = checkpoint['config']