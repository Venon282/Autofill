"""Lightning wrapper around the configurable VAE architecture."""

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pydantic import ValidationError

from src.model.configs import VAEModelConfig, VAETrainingConfig
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

        # self.save_hyperparameters({
        #     "model_config": model_config.model_dump(),
        #     "train_config": train_config.model_dump(),
        # })

        # Initialisation des poids ici
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.log_init_summary()

    def log_init_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Poids initialisés pour {self.model.__class__.__name__}")
        print(f"   - Paramètres totaux : {total_params:,}")
        print(f"   - Paramètres entraînables : {trainable_params:,}")

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
        return {**output, "data_q": getattr(self, "data_q", batch["data_q"] if "data_q" in batch else None)}

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

        # 🔹 Log LR à chaque epoch
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_epoch=True)

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
            patience=3,
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

    def get_input_dim(self):
        if hasattr(self.model, 'input_dim'):
            return self.model.input_dim
        else:
            raise AttributeError("model does not have attribute 'input_dim'.")


    def on_save_checkpoint(self, checkpoint):
        """Save a clean, reproducible state including configs, data_q, and transforms."""
        # checkpoint.clear()
        checkpoint["model_config"] = self.model_cfg.model_dump()
        checkpoint["train_config"] = self.train_cfg.model_dump()
        checkpoint["state_dict"] = self.state_dict()
        checkpoint["global_config"] = self.global_config

    def on_load_checkpoint(self, checkpoint):
        """Rebuild model and configuration from saved checkpoint, with clearer error messages."""
        try:
            model_cfg = VAEModelConfig(**checkpoint["model_config"])
        except KeyError:
            raise KeyError("Checkpoint missing key 'model_config'.")
        except ValidationError as e:
            print("Validation error in VAEModelConfig:")
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                print(f"  - {loc}: {err['msg']} ({err['type']})")
            raise

        try:
            train_cfg = VAETrainingConfig(**checkpoint["train_config"])
        except KeyError:
            raise KeyError("Checkpoint missing key 'train_config'.")
        except ValidationError as e:
            print("Validation error in VAETrainingConfig:")
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                print(f"  - {loc}: {err['msg']} ({err['type']})")
            raise

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

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
        assert model_cfg.data_q is not None, "data_q must be defined in the checkpoint's model_config."
        self.data_q = torch.tensor(model_cfg.data_q, dtype=torch.float32)

        self.transforms_data = model_cfg.transforms_data

        self.load_state_dict(checkpoint["state_dict"])

    def set_global_config(self, global_config):
        """Set global configuration for the model and submodules."""
        self.global_config = global_config
        if hasattr(self.model, 'set_global_config'):
            self.model.set_global_config(global_config)

