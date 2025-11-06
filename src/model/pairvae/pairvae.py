
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logging_utils import get_logger
from src.model.vae.pl_vae import PlVAE

logger = get_logger(__name__)


class PairVAE(nn.Module):
    """
    Compose deux VAEs pré-entraînés pour permettre la reconstruction cross-domain.
    Hypothèse: chaque sous-VAE expose au minimum:
      - __call__/forward(batch_dict) -> dict avec clés {"recon", "z"} à partir de {"data_y","data_q","metadata"}
      - decode(z) -> reconstruction
      - un attribut .model avec hyperparams utilisés pour la compatibilité (latent_dim, in_channels, etc.)
    """

    def __init__(
        self,
        vae_saxs: nn.Module = None,
        vae_les: nn.Module = None,
        ckpt_path_saxs: Union[Path, str]=None,
        ckpt_path_les: Union[Path, str]=None,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

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

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device

        ok, msg = self.check_models_compatible(raise_on_mismatch=False)
        assert ok, msg
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
            return True, "Models are None."

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
            msg = "Incompatible args" + "; ".join(diffs)
            if raise_on_mismatch:
                raise AssertionError(msg)
            return False, msg
        return True, "Models are compatible."

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Passe avant: calcule reconstructions intra-domaines et cross-domain.
        Entrée batch (tensors déjà sur device) avec clés:
          - data_y_saxs, data_q_saxs, data_y_les, data_q_les, (optionnel) metadata
        """
        metadata = batch.get("metadata", None)

        out_saxs = self.vae_saxs({
            "data_y": batch["data_y_saxs"],
            "metadata": metadata,
        })
        out_les = self.vae_les({
            "data_y": batch["data_y_les"],
            "metadata": metadata,
        })

        # cross-decode
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
