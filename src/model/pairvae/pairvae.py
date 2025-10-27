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

        ok, msg = self.check_models_compatible(raise_on_mismatch=False)
        assert ok, msg

    def forward(self, batch):
        """Return reconstructions and latent representations for paired inputs.

        inputs: dict with keys:
            - data_y_saxs: SAXS data tensor
            - data_y_les: LES data tensor
            - data_q_saxs: SAXS q-values tensor
            - data_q_les: LES q-values tensor
            - metadata: metadata dictionary

        returns: dict with keys:
        """

        metadata = batch["metadata"]
        batch_saxs = {"data_y": batch["data_y_saxs"],"data_q": batch["data_q_saxs"], "metadata": metadata}
        output_saxs = self.vae_saxs(batch_saxs)
        batch_les = {"data_y": batch["data_y_les"],"data_q": batch["data_q_les"], "metadata": metadata}
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

    def check_models_compatible(self, raise_on_mismatch=True):
        """Check if the SAXS and LES VAE architectures are compatible (ignoring weights).

        The check compares:
        - actual submodel class (type)
        - key architecture arguments from each PlVAE's config (if present):
          input_dim, latent_dim, in_channels, down_channels, up_channels,
          output_channels, strat

        Returns (bool, message). If raise_on_mismatch is True an AssertionError is raised on mismatch.
        """
        saxs_model = getattr(self.vae_saxs, "model", None)
        les_model = getattr(self.vae_les, "model", None)
        if saxs_model is None or les_model is None:
            msg = "One of the loaded PlVAE instances has no 'model' attribute."
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg

        if type(saxs_model) is not type(les_model):
            msg = f"Different model classes: {type(saxs_model).__name__} != {type(les_model).__name__}"
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg

        saxs_cfg = getattr(self.vae_saxs, "config", {}) or {}
        les_cfg = getattr(self.vae_les, "config", {}) or {}
        saxs_args = saxs_cfg.get("model", {}).get("args", {}) if isinstance(saxs_cfg, dict) else {}
        les_args = les_cfg.get("model", {}).get("args", {}) if isinstance(les_cfg, dict) else {}

        keys_to_check = [
            "input_dim",
            "latent_dim",
            "in_channels",
            "down_channels",
            "up_channels",
            "output_channels",
            "strat",
        ]

        diffs = []
        for k in keys_to_check:
            v1 = saxs_args.get(k, getattr(saxs_model, k, None))
            v2 = les_args.get(k, getattr(les_model, k, None))
            if isinstance(v1, list):
                v1_cmp = tuple(v1)
            else:
                v1_cmp = v1
            if isinstance(v2, list):
                v2_cmp = tuple(v2)
            else:
                v2_cmp = v2
            if v1_cmp != v2_cmp:
                diffs.append(f"{k}: {v1_cmp} != {v2_cmp}")

        if diffs:
            msg = "Model architecture mismatch: " + "; ".join(diffs)
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg

        return True, "Models are architecture-compatible (ignoring weights)."
