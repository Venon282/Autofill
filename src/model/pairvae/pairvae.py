from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logging_utils import get_logger

logger = get_logger(__name__)


class PairVAE(nn.Module):
    """
    Compose two VAEs to enable cross-domain reconstruction.

    It assumes each sub-VAE exposes at least:
      - __call__/forward(batch_dict) -> dict with keys {"recon", "z"}
      - decode(z) -> reconstruction
      - a `.model` attribute exposing architectural hyperparameters
        (latent_dim, in_channels, down_channels, up_channels, etc.).
    """

    def __init__(
        self,
        vae_saxs: nn.Module,
        vae_les: nn.Module,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if vae_saxs is None or vae_les is None:
            raise ValueError("Both 'vae_saxs' and 'vae_les' must be provided.")

        self.vae_saxs = vae_saxs
        self.vae_les = vae_les
        self.lr = lr

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device

        ok, msg = self.check_models_compatible(raise_on_mismatch=False)
        if not ok:
            raise AssertionError(msg)

        self.to(self.device_)

    @staticmethod
    def _get_attr_safe(obj: Any, name: str, default=None):
        return getattr(obj, name, default)

    def check_models_compatible(self, raise_on_mismatch: bool = True) -> Tuple[bool, str]:
        """
        Check if the two VAE models have compatible architectures.
        Returns (is_compatible, message).
        """
        saxs_model = self._get_attr_safe(self.vae_saxs, "model", None)
        les_model = self._get_attr_safe(self.vae_les, "model", None)
        if saxs_model is None or les_model is None:
            return True, "Models do not expose a 'model' attribute; skipping compatibility check."

        if type(saxs_model) is not type(les_model):
            msg = f"Incompatible model types: {type(saxs_model)} != {type(les_model)}"
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg

        keys = ["latent_dim", "in_channels", "down_channels", "up_channels"]
        diffs = [
            f"{k}: {self._get_attr_safe(saxs_model, k, None)} != {self._get_attr_safe(les_model, k, None)}"
            for k in keys
            if self._get_attr_safe(saxs_model, k, None) != self._get_attr_safe(les_model, k, None)
        ]
        if diffs:
            msg = "Incompatible args: " + "; ".join(diffs)
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg
        return True, "Models are compatible."

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing intra-domain and cross-domain reconstructions.

        Expected batch keys:
          - data_y_saxs, data_y_les, (optional) metadata
        """
        metadata = batch.get("metadata", None)

        out_saxs = self.vae_saxs(
            {
                "data_y": batch["data_y_saxs"],
                "metadata": metadata,
            }
        )
        out_les = self.vae_les(
            {
                "data_y": batch["data_y_les"],
                "metadata": metadata,
            }
        )

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
