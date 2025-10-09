"""Lightning module orchestrating the PairVAE training loop."""

import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.model.pairvae.loss import BarlowTwinsLoss
from src.model.pairvae.pairvae import PairVAE


class PlPairVAE(pl.LightningModule):
    """Lightning integration of the :class:`PairVAE` model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = PairVAE(self.config["model"])
        self.barlow_twins_loss = BarlowTwinsLoss(self.config["training"]["lambda_param"])
        training_cfg = self.config["training"]
        self.weight_latent_similarity = training_cfg["weight_latent_similarity"]
        self.weight_saxs2saxs = training_cfg["weight_saxs2saxs"]
        self.weight_saxs2les = training_cfg["weight_saxs2les"]
        self.weight_les2les = training_cfg["weight_les2les"]
        self.weight_les2saxs = training_cfg["weight_les2saxs"]
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)

    def compute_loss(self, batch, outputs):
        """Compute weighted reconstruction and latent alignment losses."""

        loss_saxs2saxs = F.mse_loss(outputs["recon_saxs"], batch["data_y_saxs"])
        loss_les2les = F.mse_loss(outputs["recon_les"], batch["data_y_les"])
        loss_saxs2les = F.mse_loss(outputs["recon_saxs2les"], batch["data_y_les"])
        loss_les2saxs = F.mse_loss(outputs["recon_les2saxs"], batch["data_y_saxs"])
        loss_latent = self.barlow_twins_loss(outputs["z_saxs"], outputs["z_les"])
        details = {
            "latent": loss_latent.item(),
            "saxs2saxs": loss_saxs2saxs.item(),
            "les2les": loss_les2les.item(),
            "saxs2les": loss_saxs2les.item(),
            "les2saxs": loss_les2saxs.item(),
        }
        loss_total = (
            self.weight_latent_similarity * loss_latent
            + self.weight_saxs2saxs * loss_saxs2saxs
            + self.weight_les2les * loss_les2les
            + self.weight_saxs2les * loss_saxs2les
            + self.weight_les2saxs * loss_les2saxs
        )
        return loss_total, details

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_total, details = self.compute_loss(batch, outputs)
        self.log('train_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self._log_details('train', details)
        return loss_total

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_total, details = self.compute_loss(batch, outputs)
        self.log('val_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self._log_details('val', details)

    def _log_details(self, stage: str, details):
        for key, value in details.items():
            self.log(f'{stage}_loss_{key}', value, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["max_lr"])
        warmup_epochs = self.config["training"].get("warmup_epochs", 5)
        max_epochs = self.config["training"]["num_epochs"]

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            progress = (current_epoch - warmup_epochs) / max(1, float(max_epochs - warmup_epochs))
            clipped = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * clipped))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}

    def les_to_saxs(self, batch):
        self._validate_batch(batch, 'data_y_les', 'data_q_les')
        les_batch = {
            "data_y": batch.get('data_y_les', batch.get("data_y")),
            "data_q": batch.get('data_q_les', batch.get("data_q")),
            "metadata": batch["metadata"],
        }
        output_les = self.model.vae_les(les_batch)
        recon_les2saxs = self.model.vae_saxs.decode(output_les["z"])
        return recon_les2saxs, les_batch["data_q"]

    def saxs_to_les(self, batch):
        self._validate_batch(batch, 'data_y_saxs', 'data_q_saxs')
        saxs_batch = {
            "data_y": batch.get('data_y_saxs', batch.get("data_y")),
            "data_q": batch.get('data_q_saxs', batch.get("data_q")),
            "metadata": batch["metadata"],
        }
        output_saxs = self.model.vae_saxs(saxs_batch)
        recon_saxs2les = self.model.vae_les.decode(output_saxs["z"])
        return recon_saxs2les, saxs_batch["data_q"]

    def saxs_to_saxs(self, batch):
        self._validate_batch(batch, 'data_y_saxs', 'data_q_saxs')
        saxs_batch = {
            "data_y": batch.get('data_y_saxs', batch.get("data_y")),
            "data_q": batch.get('data_q_saxs', batch.get("data_q")),
            "metadata": batch["metadata"],
        }
        output_saxs = self.model.vae_saxs(saxs_batch)
        recon_saxs2saxs = self.model.vae_saxs.decode(output_saxs["z"])
        return recon_saxs2saxs, saxs_batch["data_q"]

    def les_to_les(self, batch):
        self._validate_batch(batch, 'data_y_les', 'data_q_les')
        les_batch = {
            "data_y": batch.get('data_y_les', batch.get("data_y")),
            "data_q": batch.get('data_q_les', batch.get("data_q")),
            "metadata": batch["metadata"],
        }
        output_les = self.model.vae_les(les_batch)
        recon_les2les = self.model.vae_les.decode(output_les["z"])
        return recon_les2les, les_batch["data_q"]

    @staticmethod
    def _validate_batch(batch, y_key: str, q_key: str) -> None:
        if y_key not in batch and "data_y" not in batch:
            raise KeyError(f"Need '{y_key}' or 'data_y' in batch")
        if q_key not in batch and "data_q" not in batch:
            raise KeyError(f"Need '{q_key}' or 'data_q' in batch")

    def get_transforms_data_les(self):
        return self.model.get_les_config()["transforms_data"]

    def get_transforms_data_saxs(self):
        return self.model.get_saxs_config()["transforms_data"]
