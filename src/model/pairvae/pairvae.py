"""Wrapper combining two pretrained single-domain VAEs into a paired objective."""

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from logging_utils import get_logger
from src.model.vae.configs import VAETrainingConfig, VAEModelConfig
from src.model.vae.pl_vae import PlVAE

logger = get_logger(__name__)


class PairVAE(LightningModule):
    """Compose two pretrained VAEs to enable cross-domain reconstruction."""

    def __init__(self,
                 vae_saxs: PlVAE | None = None,
                 vae_les: PlVAE | None = None,
                 ckpt_path_saxs: str | None = None,
                 ckpt_path_les: str | None = None,
                 lr: float = 1e-4,
                 freeze_subvae: bool = False,
                 *args, **kwargs):
        """Initialize the paired VAE from pretrained single-domain VAEs.

        Args:
            vae_saxs (PlVAE | None): Preloaded SAXS VAE instance.
            vae_les (PlVAE | None): Preloaded LES VAE instance.
            ckpt_path_saxs (str | None): Optional checkpoint path for SAXS VAE.
            ckpt_path_les (str | None): Optional checkpoint path for LES VAE.
            lr (float): Learning rate for PairVAE training.
            freeze_subvae (bool): Whether to freeze sub-VAEs during training.
        """
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if vae_saxs is None:
            assert ckpt_path_saxs, ("Either a PlVAE instance or a checkpoint path must be provided for SAXS."
                                    "Check `ckpt_path_saxs` argument in config.")
            vae_saxs = PlVAE.load_from_checkpoint(ckpt_path_saxs).to(device)
            logger.info(f"Loaded SAXS VAE from checkpoint: {ckpt_path_saxs}")
        if vae_les is None:
            assert ckpt_path_les, ("Either a PlVAE instance or a checkpoint path must be provided for LES."
                                    "Check `ckpt_path_les` argument in config.")
            vae_les = PlVAE.load_from_checkpoint(ckpt_path_les).to(device)
            logger.info(f"Loaded LES VAE from checkpoint: {ckpt_path_les}")

        self.vae_saxs = vae_saxs
        self.vae_les = vae_les
        self.lr = lr
        self.freeze_subvae = freeze_subvae

        if freeze_subvae:
            for p in self.vae_saxs.parameters():
                p.requires_grad = False
            for p in self.vae_les.parameters():
                p.requires_grad = False

        ok, msg = self.check_models_compatible(raise_on_mismatch=False)
        assert ok, msg

        self.save_hyperparameters(ignore=["vae_saxs", "vae_les"])

    def forward(self, batch: dict) -> dict:
        """Perform paired forward passes and cross-domain reconstructions."""
        metadata = batch.get("metadata", None)

        out_saxs = self.vae_saxs({
            "data_y": batch["data_y_saxs"],
            "data_q": batch["data_q_saxs"],
            "metadata": metadata,
        })
        out_les = self.vae_les({
            "data_y": batch["data_y_les"],
            "data_q": batch["data_q_les"],
            "metadata": metadata,
        })

        recon_les2saxs = self.vae_saxs.decode(out_les["z"])
        recon_saxs2les = self.vae_les.decode(out_saxs["z"])

        return {
            "recon_saxs": out_saxs["recon"],
            "recon_les": out_les["recon"],
            "recon_saxs2les": recon_saxs2les,
            "recon_les2saxs": recon_les2saxs,
            "z_saxs": out_saxs["z"],
            "z_les": out_les["z"],
        }

    def training_step(self, batch, batch_idx):
        """Compute joint reconstruction and cross-domain loss."""
        out = self.forward(batch)
        loss_saxs = torch.nn.functional.mse_loss(out["recon_saxs"], batch["data_y_saxs"])
        loss_les = torch.nn.functional.mse_loss(out["recon_les"], batch["data_y_les"])
        loss_cross_1 = torch.nn.functional.mse_loss(out["recon_les2saxs"], batch["data_y_saxs"])
        loss_cross_2 = torch.nn.functional.mse_loss(out["recon_saxs2les"], batch["data_y_les"])
        total_loss = loss_saxs + loss_les + 0.5 * (loss_cross_1 + loss_cross_2)
        self.log_dict({
            "train_loss": total_loss,
            "recon_saxs": loss_saxs,
            "recon_les": loss_les,
            "cross_saxs2les": loss_cross_2,
            "cross_les2saxs": loss_cross_1,
        }, prog_bar=True, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_save_checkpoint(self, checkpoint):
        """Embed sub-VAEs' state dicts and configs in the pair checkpoint."""
        checkpoint["vae_saxs_state_dict"] = self.vae_saxs.state_dict()
        checkpoint["vae_les_state_dict"] = self.vae_les.state_dict()
        checkpoint["vae_saxs_hparams"] = getattr(self.vae_saxs, "hparams", {})
        checkpoint["vae_les_hparams"] = getattr(self.vae_les, "hparams", {})

    def on_load_checkpoint(self, checkpoint):
        """Reconstruct sub-VAEs from saved states when reloading the pair checkpoint."""

        vae_saxs_cfg = VAEModelConfig(**checkpoint["vae_saxs_hparams"]["model_config"])
        vae_saxs_train_cfg = VAETrainingConfig(**checkpoint["vae_saxs_hparams"]["train_config"])
        self.vae_saxs = PlVAE(model_config=vae_saxs_cfg, train_config=vae_saxs_train_cfg)
        self.vae_saxs.load_state_dict(checkpoint["vae_saxs_state_dict"])

        vae_les_cfg = VAEModelConfig(**checkpoint["vae_les_hparams"]["model_config"])
        vae_les_train_cfg = VAETrainingConfig(**checkpoint["vae_les_hparams"]["train_config"])
        self.vae_les = PlVAE(model_config=vae_les_cfg, train_config=vae_les_train_cfg)
        self.vae_les.load_state_dict(checkpoint["vae_les_state_dict"])


    def check_models_compatible(self, raise_on_mismatch: bool = True) -> tuple[bool, str]:
        """Ensure both submodels have identical architectural dimensions."""
        saxs_model = getattr(self.vae_saxs, "model", None)
        les_model = getattr(self.vae_les, "model", None)
        if saxs_model is None or les_model is None:
            msg = "Missing submodel in one of the VAEs."
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg
        if type(saxs_model) is not type(les_model):
            msg = f"Different model classes: {type(saxs_model).__name__} != {type(les_model).__name__}"
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg
        keys = ["latent_dim", "in_channels", "down_channels", "up_channels"]
        diffs = [f"{k}: {getattr(saxs_model, k, None)} != {getattr(les_model, k, None)}"
                 for k in keys if getattr(saxs_model, k, None) != getattr(les_model, k, None)]
        if diffs:
            msg = "Architecture mismatch: " + "; ".join(diffs)
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg
        return True, "Models are compatible."

