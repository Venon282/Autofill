"""Lightning module orchestrating the PairVAE training loop."""

import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.logging_utils import get_logger
from src.model.pairvae.loss import BarlowTwinsLoss
from src.model.pairvae.pairvae import PairVAE


logger = get_logger(__name__)


class PlPairVAE(pl.LightningModule):
    """Lightning integration of the :class:`PairVAE` model."""

    def __init__(self, config, force_dataset_q=False):
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
        self._setup_data_q_config(force_dataset_q)

    def _setup_data_q_config(self, force_dataset_q=False):
        """Configure data_q settings with warnings about source."""
        for data_type in ["saxs", "les"]:
            config_key = f"data_q_{data_type}"
            if not force_dataset_q and config_key in self.config["model"]:
                assert  self.config['model'][config_key] is not None, f"{config_key} in config cannot be None if used."
                setattr(self, config_key, self.config['model'][config_key])
                logger.warning("Using %s from configuration", config_key)
            else:
                if force_dataset_q and config_key in self.config["model"]:
                    logger.info(
                        "Forcing use of %s from dataloader (ignoring configuration value)",
                        config_key,
                    )
                else:
                    logger.warning("Using %s provided by the dataloader", config_key)

    def forward(self, batch):
        return self.model(batch)

    def compute_loss(self, batch, outputs):
        """Compute weighted reconstruction and latent alignment losses."""

        if self.config["training"]["weighted_loss"] :
            weights = torch.ones_like(outputs["recon_saxs"])         # (batch_size, 1000)
            weights[:, :self.config["training"]["weighted_loss_limit_index"]] = 10.0
            loss_saxs2saxs = (weights * (outputs["recon_saxs"] - batch["data_y_saxs"]) ** 2).sum() / weights.sum()
            loss_les2saxs = (weights * (outputs["recon_les2saxs"] - batch["data_y_saxs"]) ** 2).sum() / weights.sum()
        else :
            loss_saxs2saxs = F.mse_loss(outputs["recon_saxs"], batch["data_y_saxs"], reduction='mean')
            loss_les2saxs = F.mse_loss(outputs["recon_les2saxs"], batch["data_y_saxs"], reduction='mean')
            
        loss_les2les = F.mse_loss(outputs["recon_les"], batch["data_y_les"], reduction='mean')
        loss_saxs2les = F.mse_loss(outputs["recon_saxs2les"], batch["data_y_les"], reduction='mean')

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

    def _prepare_batch(self, batch, data_type):
        """Prepare batch for specific data type, using config data_q if available."""
        y_key = f'data_y_{data_type}'
        q_key = f'data_q_{data_type}'

        self._validate_batch(batch, y_key)

        # Use config data_q if defined, otherwise fallback to batch data_q
        config_q_attr = f'data_q_{data_type}'
        if hasattr(self, config_q_attr):
            data_q = getattr(self, config_q_attr)
        else:
            data_q = batch.get(q_key, batch.get("data_q"))

        return {
            "data_y": batch.get(y_key, batch.get("data_y")),
            "data_q": data_q,
            "metadata": batch["metadata"],
        }, data_q

    def les_to_saxs(self, batch):
        les_batch, _ = self._prepare_batch(batch, 'les')
        output_les = self.model.vae_les(les_batch)
        recon_les2saxs = self.model.vae_saxs.decode(output_les["z"])
        return recon_les2saxs, self.get_data_q_saxs() if hasattr(self, 'data_q_saxs') else data_q
    def saxs_to_les(self, batch):
        saxs_batch, _ = self._prepare_batch(batch, 'saxs')
        output_saxs = self.model.vae_saxs(saxs_batch)
        recon_saxs2les = self.model.vae_les.decode(output_saxs["z"])
        return recon_saxs2les, self.get_data_q_les() if hasattr(self, 'data_q_les')else data_q

    def saxs_to_saxs(self, batch):
        saxs_batch, data_q = self._prepare_batch(batch, 'saxs')
        output_saxs = self.model.vae_saxs(saxs_batch)
        recon_saxs2saxs = self.model.vae_saxs.decode(output_saxs["z"])
        return recon_saxs2saxs, self.get_data_q_saxs() if hasattr(self, 'data_q_saxs') else data_q

    def les_to_les(self, batch):
        les_batch, data_q = self._prepare_batch(batch, 'les')
        output_les = self.model.vae_les(les_batch)
        recon_les2les = self.model.vae_les.decode(output_les["z"])
        return recon_les2les, self.get_data_q_les() if hasattr(self, 'data_q_les') else data_q

    @staticmethod
    def _validate_batch(batch, y_key: str) -> None:
        if y_key not in batch and "data_y" not in batch:
            raise KeyError(f"Need '{y_key}' or 'data_y' in batch")

    def get_transforms_data_les(self):
        return self.model.get_les_config()["transforms_data"]

    def get_transforms_data_saxs(self):
        return self.model.get_saxs_config()["transforms_data"]

    # def get_data_q_saxs(self):
    #     """Return the original data_q_saxs array"""
    #     return self.data_q_saxs

    # def get_data_q_les(self):
    #     """Return the original data_q_les array"""
    #     return self.data_q_les

    def get_data_q_saxs(self):
        if hasattr(self, 'data_q_saxs'):
            return self.data_q_saxs
        else:
            raise AttributeError("data_q_saxs is not set. Please ensure it is provided in the dataloader or config.")

    def get_data_q_les(self):
        if hasattr(self, 'data_q_les'):
            return self.data_q_les
        else:
            raise AttributeError("data_q_les is not set. Please ensure it is provided in the dataloader or config.")