"""Wrapper combining two single-domain VAEs into a paired objective."""

from typing import Optional

import torch
import torch.nn as nn

from src.model.vae.pl_vae import PlVAE


class PairVAE(nn.Module):
    """Compose SAXS and LES VAEs to enable cross-reconstruction."""

    def __init__(
        self,
        config,
        *,
        load_pretrained_from_path: Optional[bool] = None,
        map_location=None,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._map_location = map_location if map_location is not None else self.device
        self.vae_saxs, self._vae_saxs_config = self._load_subvae(
            domain="saxs",
            sub_config=self.config["VAE_SAXS"],
            load_from_path=load_pretrained_from_path,
        )
        self.vae_les, self._vae_les_config = self._load_subvae(
            domain="les",
            sub_config=self.config["VAE_LES"],
            load_from_path=load_pretrained_from_path,
        )

        # Persist resolved sub-VAE configuration in the hyperparameters so that
        # checkpoints remain self-contained and can be reloaded on a different
        # machine without the original checkpoint paths.
        self.config["VAE_SAXS"]["config"] = self._vae_saxs_config
        self.config["VAE_LES"]["config"] = self._vae_les_config

        print("_vae_saxs_config: ", self.config["VAE_SAXS"])
        print("_vae_les_config: ", self.config["VAE_LES"])

        ok, msg = self.check_models_compatible(raise_on_mismatch=False)
        assert ok, msg

    def _load_subvae(
        self,
        domain: str,
        sub_config: dict,
        load_from_path: Optional[bool],
    ):
        """Instantiate one of the constituent VAEs.

        Parameters
        ----------
        domain:
            Either ``"saxs"`` or ``"les"``.
        sub_config:
            Configuration dictionary taken from ``self.config`` for the
            corresponding VAE. It may contain the keys ``path_checkpoint`` and
            ``config``.
        load_from_path:
            Whether to attempt loading weights from ``path_checkpoint``. During
            PairVAE training this is ``True`` so we can bootstrap the model from
            separately trained VAEs. When ``None`` the loader infers the
            behaviour automatically: it loads from the checkpoint path only if
            a path is provided *and* no stored configuration is available in the
            PairVAE checkpoint. When restoring a PairVAE checkpoint for
            inference we therefore default to using the stored configuration,
            avoiding dependence on file paths that may no longer exist.
        """

        checkpoint_path = sub_config.get("path_checkpoint")
        stored_config = sub_config.get("config")
        was_resolved = bool(stored_config.get("resolved", False))

        if load_from_path is None:
            if was_resolved:
                # When the PairVAE is restored from one of its own checkpoints we
                # prefer using the stored configuration instead of reloading the
                # constituent VAEs from potentially stale filesystem paths.
                load_from_path = False
            else:
                # During fresh training runs we expect valid checkpoint paths in
                # the configuration so that the pretrained VAEs can be used to
                # initialise the PairVAE weights.
                load_from_path = bool(checkpoint_path)

        if load_from_path and checkpoint_path:
            vae = PlVAE.load_from_checkpoint(checkpoint_path, map_location=self._map_location).to(self.device)
            vae_config = getattr(vae, "config", None)
            if vae_config is None:
                raise ValueError(f"Loaded {domain.upper()} VAE from '{checkpoint_path}' without configuration")
        else:
            if stored_config is None:
                raise ValueError(
                    "Cannot instantiate {dom} VAE without a stored configuration. "
                    "Ensure the PairVAE checkpoint was saved with sub-vae configs or provide valid paths."
                    .format(dom=domain.upper())
                )
            vae = PlVAE(stored_config).to(self.device)
            vae_config = stored_config

        vae_config.update(
            {
                "resolved": True,
                "last_source": "path" if load_from_path and checkpoint_path else "config",
            }
        )

        return vae, vae_config

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
