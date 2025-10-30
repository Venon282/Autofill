"""Lightning wrapper around the configurable VAE architecture."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model.vae.configs import VAEModelConfig, VAETrainingConfig
from src.logging_utils import get_logger
from src.model.vae.submodel.registry import *


logger = get_logger(__name__)


class PlVAE(pl.LightningModule):
    """Lightning wrapper around a configurable single-domain variational auto-encoder."""

    def __init__(
        self,
        model_config: VAEModelConfig = None,
        train_config: VAETrainingConfig = None,
        force_dataset_q: bool = False,
    ):
        super().__init__()
        if model_config is None or train_config is None:
            self.model_cfg = None
            self.train_cfg = None
            self.model = None
            self.beta = 0.0
            self._lazy_initialized = True
            return
        model_config, train_config = self.safe_check_config(model_config, train_config)

        self.model_cfg = model_config
        self.train_cfg = train_config
        self.beta = model_config.beta
        self.transforms_data = self.model_cfg.transforms_data
        self.data_q = self.model_cfg.data_q

        model_class = MODEL_REGISTRY.get(model_config.vae_class)
        if model_class is None:
            raise ValueError(
                f"Unknown VAE class '{model_config.vae_class}' "
                f"(expected one of {list(MODEL_REGISTRY.keys())})."
            )

        try:
            self.model = model_class(**model_config.args)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate model '{model_config.vae_class}' "
                f"with arguments from 'model.args'.\n\n"
                f"Original error: {e}\n\n"
                f"Likely cause: one or more parameters are missing or invalid in the configuration.\n"
                f"Check your configuration path: 'model.args'\n"
                f"Provided arguments:\n{model_config.args}"
            ) from e

        if not force_dataset_q and model_config.data_q is not None:
            self.data_q = torch.tensor(model_config.data_q, dtype=torch.float32)
            logger.warning("Using data_q from configuration.")
        else:
            logger.warning("Using data_q provided by the dataloader.")

        # self.save_hyperparameters({
        #     "model_config": model_config.model_dump(),
        #     "train_config": train_config.model_dump(),
        # })

    def safe_check_config(self, model_config: VAEModelConfig, train_config: VAETrainingConfig) -> tuple[
        VAEModelConfig, VAETrainingConfig]:
        """Ensure model_config and train_config are of correct types."""
        if not isinstance(model_config, VAEModelConfig) and model_config is not None:
            try:
                model_config = VAEModelConfig(**model_config.model_dump())
            except Exception as e:
                raise TypeError(
                    f"model_config must be a VAEModelConfig or a compatible dict-like object.\n"
                    f"Original error: {e}"
                ) from e
        if not isinstance(train_config, VAETrainingConfig) and model_config is not None:
            try:
                train_config = VAETrainingConfig(**train_config.model_dump())
            except Exception as e:
                raise TypeError(
                    f"train_config must be a VAETrainingConfig or a compatible dict-like object.\n"
                    f"Original error: {e}"
                ) from e
        return model_config, train_config

    def forward(self, batch):
        output = self.model(x=batch["data_y"], metadata=batch["metadata"])
        return {**output, "data_q": getattr(self, "data_q", batch["data_q"])}

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def compute_loss(self, batch, output):
        """Return VAE loss components with optional weighted reconstruction."""
        recon, target = output["recon"], batch["data_y"]
        mu, logvar = output["mu"], output["logvar"]
        if self.train_cfg.weighted_loss:
            weights = torch.ones_like(target)
            limit = self.train_cfg.weighted_loss_limit_index
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

    def test_step(self, batch, batch_idx):
        """Compute test metrics."""

        output = self.model(x=batch["data_y"], metadata=batch["metadata"])
        loss, recon_loss, kl_loss = self.compute_loss(batch, output)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('test_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        """Use AdamW with a ReduceLROnPlateau scheduler on validation loss."""

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_cfg.max_lr)
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

    def get_transformer_q(self):
        if hasattr(self, 'transforms_data'):
            return self.transforms_data['q']
        else:
            raise AttributeError("transforms_data is not set in PlVAE.")

    def get_transformer_y(self):
        if hasattr(self, 'transforms_data'):
            return self.transforms_data['y']
        else:
            raise AttributeError("transforms_data is not set in PlVAE.")

    def get_transformer(self):
        if hasattr(self, 'transforms_data'):
            return self.transforms_data
        else:
            raise AttributeError("transforms_data is not set in PlVAE.")


    def on_save_checkpoint(self, checkpoint):
        """Save a clean, reproducible state including configs, data_q, and transforms."""
        # checkpoint.clear()
        checkpoint["model_config"] = self.model_cfg.model_dump()
        checkpoint["train_config"] = self.train_cfg.model_dump()
        checkpoint["state_dict"] = self.state_dict()
        checkpoint["global_config"] = self.global_config

    def on_load_checkpoint(self, checkpoint):
        """Rebuild model and configuration from saved checkpoint."""
        model_cfg = VAEModelConfig(**checkpoint["model_config"])
        train_cfg = VAETrainingConfig(**checkpoint["train_config"])

        model_class = MODEL_REGISTRY.get(model_cfg.vae_class)
        if model_class is None:
            raise ValueError(
                f"Unknown VAE class '{model_cfg.vae_class}' "
                f"(expected one of {list(MODEL_REGISTRY.keys())})."
            )

        self.model = model_class(**model_cfg.args)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.beta = model_cfg.beta
        self.set_global_config(checkpoint['global_config'])

        self.data_q = torch.tensor(model_cfg.data_q, dtype=torch.float32)
        self.transforms_data = model_cfg.transforms_data

        self.load_state_dict(checkpoint["state_dict"])

    def set_global_config(self, global_config):
        """Set global configuration for the model and submodules."""
        self.global_config = global_config
        if hasattr(self.model, 'set_global_config'):
            self.model.set_global_config(global_config)

