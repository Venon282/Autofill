"""Lightning module orchestrating the PairVAE training loop."""

from typing import Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from src.model.configs import (
    VAEModelConfig,
    VAETrainingConfig,
    PairVAEModelConfig,
    PairVAETrainingConfig,
    ModelSpec,
)
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

        self.spec = ModelSpec.PAIR
        if isinstance(model_config, dict):
            model_config = PairVAEModelConfig(**model_config)
        if isinstance(train_config, dict):
            train_config = PairVAETrainingConfig(**train_config)
        self.model_cfg = model_config
        self.train_cfg = train_config

        # The actual PairVAE will be attached either:
        # - by a factory method at training time, or
        # - inside `on_load_checkpoint` when restoring from a checkpoint.
        self.model: PairVAE | None = None

        self.barlow_twins_loss = BarlowTwinsLoss(lambda_coeff=train_config.lambda_param)

        self.save_hyperparameters(
            {
                "model_config": model_config.model_dump(),
                "train_config": train_config.model_dump(),
            }
        )

    @classmethod
    def from_pretrained_subvaes(
        cls,
        model_config: PairVAEModelConfig,
        train_config: PairVAETrainingConfig,
        map_location: Union[str, torch.device, None] = None,
    ) -> "PlPairVAE":
        """Build a PlPairVAE from two pretrained PlVAE checkpoints."""
        from src.model.vae.pl_vae import PlVAE
        ckpt_path_saxs = model_config.ckpt_path_saxs
        ckpt_path_les = model_config.ckpt_path_les

        vae_saxs = PlVAE.load_from_checkpoint(ckpt_path_saxs, map_location=map_location)
        vae_les = PlVAE.load_from_checkpoint(ckpt_path_les, map_location=map_location)

        instance = cls(model_config=model_config, train_config=train_config)

        lr = getattr(model_config, "lr", None)
        if lr is None:
            lr = train_config.max_lr

        instance.model = PairVAE(
            vae_saxs=vae_saxs,
            vae_les=vae_les,
            lr=lr,
            device=instance.device,
        )
        return instance

    def forward(self, batch):
        if self.model is None:
            raise RuntimeError("PairVAE model is not initialized.")
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

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_epoch=True)

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
        if self.model is None:
            raise RuntimeError("PairVAE model is not initialized.")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_cfg.max_lr)

        from src.model.pairvae.scheduler import WarmupLR

        scheduler = WarmupLR(
            optimizer,
            warmup_epochs=self.train_cfg.warmup_epochs,
            max_lr=self.train_cfg.max_lr,
            eta_min=self.train_cfg.eta_min,
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
        y_key = f"data_y_{data_type}"

        self._validate_batch(batch, y_key)


        return {
            "data_y": batch.get(y_key, batch.get("data_y")),
            "metadata": batch["metadata"],
        }

    def les_to_saxs(self, batch):
        with torch.no_grad():
            les_batch = self._prepare_batch(batch, 'les')
            output_les = self.model.vae_les(les_batch)
            recon_les2saxs = self.model.vae_saxs.decode(output_les["z"])
        return recon_les2saxs, self.get_data_q_saxs() if hasattr(self, 'data_q_saxs') else output_les['data_q']

    def saxs_to_les(self, batch):
        with torch.no_grad():
            saxs_batch = self._prepare_batch(batch, 'saxs')
            output_saxs = self.model.vae_saxs(saxs_batch)
            recon_saxs2les = self.model.vae_les.decode(output_saxs["z"])
        return recon_saxs2les, self.get_data_q_les() if hasattr(self, 'data_q_les') else output_saxs['data_q']

    def saxs_to_saxs(self, batch):
        with torch.no_grad():
            saxs_batch = self._prepare_batch(batch, 'saxs')
            output_saxs = self.model.vae_saxs(saxs_batch)
            recon_saxs2saxs = self.model.vae_saxs.decode(output_saxs["z"])
        return recon_saxs2saxs, self.get_data_q_saxs() if hasattr(self, 'data_q_saxs') else output_saxs['data_q']

    def les_to_les(self, batch):
        with torch.no_grad():
            les_batch = self._prepare_batch(batch, 'les')
            output_les = self.model.vae_les(les_batch)
            recon_les2les = self.model.vae_les.decode(output_les["z"])
        return recon_les2les, self.get_data_q_les() if hasattr(self, 'data_q_les') else output_les['data_q']

    @staticmethod
    def _validate_batch(batch, y_key: str) -> None:
        if y_key not in batch and "data_y" not in batch:
            raise KeyError(f"Need '{y_key}' or 'data_y' in batch")

    def get_transforms_data_les(self):
        if self.model is None:
            raise RuntimeError("PairVAE model is not initialized.")
        return self.model.vae_les.get_transformer()

    def get_transforms_data_saxs(self):
        if self.model is None:
            raise RuntimeError("PairVAE model is not initialized.")
        return self.model.vae_saxs.get_transformer()

    def get_data_q_saxs(self):
        if self.model is None:
            raise RuntimeError("PairVAE model is not initialized.")
        return self.model.vae_saxs.get_data_q()

    def get_data_q_les(self):
        if self.model is None:
            raise RuntimeError("PairVAE model is not initialized.")
        return self.model.vae_les.get_data_q()

    def on_load_checkpoint(self, checkpoint):
        """Restore PairVAE and its sub-VAEs from unified checkpoint metadata."""
        try:
            self.model_cfg = PairVAEModelConfig(**checkpoint["model_config"])
        except KeyError:
            try:
                self.model_cfg = PairVAEModelConfig(**checkpoint["pairvae_model_config"])
            except KeyError:
                raise KeyError("Checkpoint must contain either 'model_config' or 'pairvae_model_config'")

        try:
            self.train_cfg = PairVAETrainingConfig(**checkpoint["train_config"])
        except KeyError:
            try:
                self.train_cfg = PairVAETrainingConfig(**checkpoint["pairvae_train_config"])
            except KeyError:
                raise KeyError("Checkpoint must contain either 'train_config' or 'pairvae_train_config'")

        def _restore_vae(entry):
            from src.model.vae.pl_vae import PlVAE
            try:
                model_cfg = VAEModelConfig(**entry["vae_model_config"])
            except KeyError:
                 model_cfg = VAEModelConfig(**entry["model_config"])
            try:
                train_cfg = VAETrainingConfig(**entry["vae_train_config"])
            except KeyError:
                train_cfg = VAETrainingConfig(**entry["train_config"])
            vae = PlVAE(model_config=model_cfg, train_config=train_cfg)
            return vae

        vae_saxs = _restore_vae(checkpoint["vae_saxs"])
        vae_les = _restore_vae(checkpoint["vae_les"])

        lr = getattr(self.model_cfg, "lr", None)
        if lr is None:
            lr = self.train_cfg.max_lr

        self.model = PairVAE(
            vae_saxs=vae_saxs,
            vae_les=vae_les,
            lr=lr,
            device=self.device,
        )

        global_config = checkpoint.get("global_config")
        if global_config is not None:
            self.set_global_config(global_config)

    def on_save_checkpoint(self, checkpoint):
        """Save sub-VAEs configuration for reconstruction on load."""
        checkpoint["model_config"] = self.model_cfg.model_dump()
        checkpoint["train_config"] = self.train_cfg.model_dump()

        if hasattr(self, "global_config"):
            checkpoint["global_config"] = self.global_config
        else:
            checkpoint["global_config"] = None
            logger.warning("No global_config found in PlPairVAE during checkpoint saving.")

        if self.model is None:
            logger.warning("PlPairVAE.on_save_checkpoint called with model=None.")
            return

        def _extract_vae_info(vae):
            return {
                "model_config": vae.model_cfg.model_dump(),
                "train_config": vae.train_cfg.model_dump(),
            }

        checkpoint["vae_saxs"] = _extract_vae_info(self.model.vae_saxs)
        checkpoint["vae_les"] = _extract_vae_info(self.model.vae_les)

    def set_global_config(self, global_config):
        """Set global configuration for the model and submodules."""
        self.global_config = global_config
        if self.model is not None and hasattr(self.model, "set_global_config"):
            self.model.set_global_config(global_config)
