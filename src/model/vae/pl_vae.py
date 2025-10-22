"""Lightning wrapper around the configurable VAE architecture."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model.vae.submodel.registry import *


class PlVAE(pl.LightningModule):
    """Train and evaluate variational auto-encoders using Lightning."""

    def __init__(self, config, force_dataset_q=False):
        super().__init__()
        if config is None and not hasattr(self, 'config'):
            raise ValueError("Configuration dictionary is required for PlVAE.")
        # print(f"[PlVAE] Initializing with config: {config}")
        if config is not None:
            self.config = config
        self.beta = config["training"]["beta"]
        model_class = self.config["model"]["vae_class"]
        self.model = MODEL_REGISTRY.get(model_class)(**self.config["model"]["args"])
        if not force_dataset_q and "data_q" in self.config["model"]:
            setattr(self, "data_q", self.config["model"]["data_q"])
            print(f"[PlVAE] WARNING: Using data_q from config, not from dataloader!")
        else:
            if force_dataset_q and "data_q" in self.config["model"]:
                print(f"[PlVAE] INFO: Forcing use of data_q from dataloader (ignoring config)")
            else:
                print(f"[PlVAE] WARNING: Using data_q from dataloader, not from config!")

    def forward(self, batch):
        """Forward pass delegating to the configured sub-model."""
        return self.model(x=batch["data_y"], metadata=batch["metadata"]) | {"data_q" : self.data_q if hasattr(self, 'data_q') else batch["data_q"]}

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def compute_loss(self, batch, output):
        """Return VAE loss components with optional weighted reconstruction."""

        target = batch["data_y"]
        recon = output['recon']
        mu = output['mu']
        logvar = output['logvar']
        if self.config["training"]["weighted_loss"]:
            weights = torch.ones_like(target)
            limit = self.config["training"]["weighted_loss_limit_index"]
            weights[:, :limit] = 10.0
            recon_loss = (weights * (recon - target) ** 2).sum() / weights.sum()
        else:
            recon_loss = F.mse_loss(recon, target, reduction='mean')
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + self.beta * kl_div, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        """Compute training losses and log metrics."""

        output = self.model(x=batch["data_y"], metadata=batch["metadata"])
        loss, recon_loss, kl_loss = self.compute_loss(batch, output)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation metrics."""

        output = self.model(x=batch["data_y"], metadata=batch["metadata"])
        loss, recon_loss, kl_loss = self.compute_loss(batch, output)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        """Use AdamW with a ReduceLROnPlateau scheduler on validation loss."""

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["max_lr"])
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            threshold=1e-3,
            factor=0.1,
            patience=5,
            min_lr=1e-10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def get_data_q(self):
        if hasattr(self, 'data_q'):
            return self.data_q
        else:
            raise AttributeError("data_q is not set in PlVAE.")