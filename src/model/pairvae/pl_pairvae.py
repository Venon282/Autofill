"""Lightning module orchestrating the PairVAE training loop."""

import math
from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from src.model.pairvae.configs import PairVAEModelConfig, PairVAETrainingConfig
from src.model.vae.configs import VAEModelConfig, VAETrainingConfig
from src.logging_utils import get_logger
from src.model.pairvae.loss import BarlowTwinsLoss
from src.model.pairvae.pairvae import PairVAE


logger = get_logger(__name__)


class PlPairVAE(pl.LightningModule):
    """Lightning integration of the PairVAE model."""

    def __init__(
        self,
        model_config: PairVAEModelConfig,
        train_config: PairVAETrainingConfig,
    ):
        super().__init__()
        self.model_cfg = model_config
        self.train_cfg = train_config

        self.model = PairVAE(**model_config.model_dump())

        self.barlow_twins_loss = BarlowTwinsLoss(lambda_coeff=train_config.lambda_param)
        # self._setup_data_q_config(force_dataset_q)
        self.save_hyperparameters({
            "model_config": model_config.model_dump(),
            "train_config": train_config.model_dump()
        })

    # def _setup_data_q_config(self, force_dataset_q=False):
    #     for data_type in ["saxs", "les"]:
    #         config_key = f"data_q_{data_type}"
    #         value = getattr(self.model_cfg, config_key)
    #         if not force_dataset_q and value is not None:
    #             setattr(self, config_key, torch.tensor(value, device=self.device))
    #             logger.warning("Using %s from configuration", config_key)
    #         else:
    #             logger.warning("Using %s provided by the dataloader", config_key)

    def forward(self, batch):
        return self.model(batch)

    def compute_loss(self, batch, outputs):
        weighted_loss = self.train_cfg.weighted_loss
        weighted_limit = self.train_cfg.weighted_loss_limit_index

        if weighted_loss and weighted_limit is not None:
            weights = torch.ones_like(outputs["recon_saxs"])
            weights[:, :weighted_limit] = 10.0
            loss_saxs2saxs = (weights * (outputs["recon_saxs"] - batch["data_y_saxs"]) ** 2).sum() / weights.sum()
            loss_les2saxs = (weights * (outputs["recon_les2saxs"] - batch["data_y_saxs"]) ** 2).sum() / weights.sum()
        else:
            loss_saxs2saxs = F.mse_loss(outputs["recon_saxs"], batch["data_y_saxs"])
            loss_les2saxs = F.mse_loss(outputs["recon_les2saxs"], batch["data_y_saxs"])

        loss_les2les = F.mse_loss(outputs["recon_les"], batch["data_y_les"])
        loss_saxs2les = F.mse_loss(outputs["recon_saxs2les"], batch["data_y_les"])
        loss_latent = self.barlow_twins_loss(outputs["z_saxs"], outputs["z_les"])

        details = {
            "latent": loss_latent.item(),
            "saxs2saxs": loss_saxs2saxs.item(),
            "les2les": loss_les2les.item(),
            "saxs2les": loss_saxs2les.item(),
            "les2saxs": loss_les2saxs.item(),
        }

        total = (
            self.train_cfg.weight_latent_similarity * loss_latent
            + self.train_cfg.weight_saxs2saxs * loss_saxs2saxs
            + self.train_cfg.weight_les2les * loss_les2les
            + self.train_cfg.weight_saxs2les * loss_saxs2les
            + self.train_cfg.weight_les2saxs * loss_les2saxs
        )
        return total, details

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_total, details = self.compute_loss(batch, outputs)
        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self._log_details("train", details)
        return loss_total

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_total, details = self.compute_loss(batch, outputs)
        self.log("val_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self._log_details("val", details)

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_total, details = self.compute_loss(batch, outputs)
        self.log("test_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self._log_details("test", details)

    def _log_details(self, stage: str, details):
        for k, v in details.items():
            self.log(f"{stage}_loss_{k}", v, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        """Use AdamW with a ReduceLROnPlateau scheduler on validation loss."""

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_cfg.max_lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            threshold=1e-3,
            factor=0.1,
            patience=10,
            min_lr=self.train_cfg.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

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
        }

    def les_to_saxs(self, batch):
        with torch.no_grad():
            les_batch, _ = self._prepare_batch(batch, 'les')
            output_les = self.model.vae_les(les_batch)
            recon_les2saxs = self.model.vae_saxs.decode(output_les["z"])
        return recon_les2saxs, self.get_data_q_saxs() if hasattr(self, 'data_q_saxs') else output_les['data_q']

    def saxs_to_les(self, batch):
        with torch.no_grad():
            saxs_batch, _ = self._prepare_batch(batch, 'saxs')
            output_saxs = self.model.vae_saxs(saxs_batch)
            recon_saxs2les = self.model.vae_les.decode(output_saxs["z"])
        return recon_saxs2les, self.get_data_q_les() if hasattr(self, 'data_q_les') else output_saxs['data_q']

    def saxs_to_saxs(self, batch):
        with torch.no_grad():
            saxs_batch, data_q = self._prepare_batch(batch, 'saxs')
            output_saxs = self.model.vae_saxs(saxs_batch)
            recon_saxs2saxs = self.model.vae_saxs.decode(output_saxs["z"])
        return recon_saxs2saxs, self.get_data_q_saxs() if hasattr(self, 'data_q_saxs') else output_saxs['data_q']

    def les_to_les(self, batch):
        with torch.no_grad():
            les_batch, data_q = self._prepare_batch(batch, 'les')
            output_les = self.model.vae_les(les_batch)
            recon_les2les = self.model.vae_les.decode(output_les["z"])
        return recon_les2les, self.get_data_q_les() if hasattr(self, 'data_q_les') else output_les['data_q']

    @staticmethod
    def _validate_batch(batch, y_key: str) -> None:
        if y_key not in batch and "data_y" not in batch:
            raise KeyError(f"Need '{y_key}' or 'data_y' in batch")

    def get_transforms_data_les(self):
        return self.model.vae_les.get_transformer()

    def get_transforms_data_saxs(self):
        return self.model.vae_saxs.get_transformer()

    # def get_data_q_saxs(self):
    #     """Return the original data_q_saxs array"""
    #     return self.data_q_saxs

    # def get_data_q_les(self):
    #     """Return the original data_q_les array"""
    #     return self.data_q_les

    def get_data_q_saxs(self):
        return self.model.vae_saxs.get_data_q()

    def get_data_q_les(self):
        return self.model.vae_les.get_data_q()

    # def have_data_q_saxs(self):
    #     return hasattr(self, 'data_q_saxs')
    #
    # def have_data_q_les(self):
    #     return hasattr(self, 'data_q_les')

    def on_load_checkpoint(self, checkpoint):
        """Restore full PairVAE and its sub-VAEs from unified checkpoint."""
        pair_model_cfg = PairVAEModelConfig(**checkpoint["pairvae_model_config"])
        pair_train_cfg = PairVAETrainingConfig(**checkpoint["pairvae_train_config"])
        self.model_cfg = pair_model_cfg
        self.train_cfg = pair_train_cfg

        def _restore_vae(entry):
            from src.model.vae.pl_vae import PlVAE
            model_cfg = VAEModelConfig(**entry["model_config"])
            train_cfg = VAETrainingConfig(**entry["train_config"])
            vae = PlVAE(model_config=model_cfg,
                        train_config=train_cfg)
            return vae

        self.model.vae_saxs = _restore_vae(checkpoint["vae_saxs"])
        self.model.vae_les = _restore_vae(checkpoint["vae_les"])
        self.set_global_config(checkpoint['global_config'])

        self.load_state_dict(checkpoint["state_dict"])

    def on_save_checkpoint(self, checkpoint):
        """Save both sub-VAEs and their full configuration."""
        # checkpoint.clear()

        checkpoint["pairvae_model_config"] = self.model_cfg.model_dump()
        checkpoint["pairvae_train_config"] = self.train_cfg.model_dump()
        checkpoint["state_dict"] = self.state_dict()
        checkpoint["global_config"] = self.global_config

        def _extract_vae_info(vae):
            return {
                "model_config": vae.model_cfg.model_dump(),
                "train_config": vae.train_cfg.model_dump(),
                "state_dict": vae.state_dict(),
                "data_q": getattr(vae, "data_q", []).tolist(),
                "transforms_data": getattr(vae, "transforms_data", {}),
            }

        checkpoint["vae_saxs"] = _extract_vae_info(self.model.vae_saxs)
        checkpoint["vae_les"] = _extract_vae_info(self.model.vae_les)


    def set_global_config(self, global_config):
        """Set global configuration for the model and submodules."""
        self.global_config = global_config
        if hasattr(self.model, 'set_global_config'):
            self.model.set_global_config(global_config)
