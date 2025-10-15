"""Wrapper combining two single-domain VAEs into a paired objective."""

import torch
import torch.nn as nn

from src.model.vae.pl_vae import PlVAE


class PairVAE(nn.Module):
    """Compose SAXS and LES VAEs to enable cross-reconstruction."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_saxs = PlVAE.load_from_checkpoint(self.config["VAE_SAXS"]["path_checkpoint"]).to(self.device)
        self._vae_saxs_config = self.vae_saxs.config
        self.vae_les = PlVAE.load_from_checkpoint(self.config["VAE_LES"]["path_checkpoint"]).to(self.device)
        self._vae_les_config = self.vae_les.config

    def forward(self, batch):
        """Return reconstructions and latent representations for paired inputs."""

        metadata = batch["metadata"]
        batch_saxs = {"data_y": batch["data_y_saxs"], "metadata": metadata}
        output_saxs = self.vae_saxs(batch_saxs)
        batch_les = {"data_y": batch["data_y_les"], "metadata": metadata}
        output_les = self.vae_les(batch_les)
        recon_les2saxs = self.vae_saxs.decode(output_les["z"])
        recon_saxs2les = self.vae_les.decode(output_saxs["z"])
        return {
            "recon_saxs": output_saxs["recon"],
            "recon_les": output_les["recon"],
            "recon_saxs2les": recon_saxs2les,
            "recon_les2saxs": recon_les2saxs,
            "z_saxs": output_saxs["z"],
            "z_les": output_les["z"],
        }

    def get_les_config(self):
        """Return the configuration used to train the LES VAE."""

        return self._vae_les_config

    def get_saxs_config(self):
        """Return the configuration used to train the SAXS VAE."""

        return self._vae_saxs_config
