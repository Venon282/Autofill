from __future__ import annotations

import math
import os
from random import sample
from typing import Any, Dict, Iterable, Sequence, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import numpy as np

from src.logging_utils import get_logger


logger = get_logger(__name__)

"""Lightning callback utilities for running inference plots during validation."""

TensorBatch = Union[torch.Tensor, Dict[str, Any], Iterable[Any]]


def move_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """Recursively send nested tensors and collections to ``device``."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


class InferencePlotCallback(pl.Callback):
    """Generate and log diagnostic plots from validation batches."""

    def __init__(
        self,
        curves_config: Dict[str, Dict[str, Any]],
        artifact_file: str = "plot.png",
        output_dir: str | None = None,
        num_samples: int = 10,
        every_n_epochs: int = 5,
    ) -> None:
        self.curves_config = curves_config
        self._validate_config()
        self.base_artifact_file = artifact_file
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run inference every ``every_n_epochs`` epochs."""

        if trainer.current_epoch % self.every_n_epochs == 0:
            self._infer_and_plot(trainer, pl_module)

    def _validate_config(self) -> None:
        """Ensure the curve configuration dictionary is well-formed."""

        if not isinstance(self.curves_config, dict):
            raise ValueError("curves_config must be a dictionary")
        for name, cfg in self.curves_config.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Configuration for {name} must be a dictionary")
            if 'truth_key' not in cfg or 'pred_keys' not in cfg:
                raise ValueError(f"Configuration for {name} must contain 'truth_key' and 'pred_keys'")
            if not isinstance(cfg['pred_keys'], list):
                try:
                    cfg['pred_keys'] = list(cfg['pred_keys'])
                except TypeError as exc:
                    raise ValueError(f"'pred_keys' for {name} must be a list") from exc

    def _infer_and_plot(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        """Fetch a validation batch, run inference, and log plots."""

        model.eval()
        dataloaders = trainer.val_dataloaders
        dataloader: Sequence[Any]
        if isinstance(dataloaders, (list, tuple)):
            dataloader = dataloaders[0]
        else:
            dataloader = dataloaders
        batch = next(iter(dataloader))
        batch = move_to_device(batch, model.device)
        with torch.inference_mode():
            raw_outputs = model(batch)
            for name, cfg in self.curves_config.items():
                truth = batch[cfg['truth_key']]
                if isinstance(truth, torch.Tensor) and truth.ndim > 1:
                    truth = truth.squeeze(1)
                preds: Dict[str, torch.Tensor] = {}
                for key in cfg['pred_keys']:
                    value = raw_outputs[key]
                    if isinstance(value, torch.Tensor) and value.ndim > 1:
                        value = value.squeeze(1)
                    preds[key] = value
                self._plot(trainer, name, truth, preds, cfg.get('use_loglog', False))
        model.train()

    def _plot(
        self,
        trainer: pl.Trainer,
        name: str,
        truth: torch.Tensor,
        preds: Dict[str, torch.Tensor],
        use_loglog: bool,
    ) -> None:
        """Render and log comparison plots for a single set of curves."""

        sample_count = min(self.num_samples, len(truth))
        if sample_count == 0:
            return
        indices = sample(range(len(truth)), sample_count)
        n_rows = math.ceil(sample_count / 2)
        fig, axs = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
        axes = np.array(axs).reshape(-1)
        for row, index in enumerate(indices):
            ax = axes[row]
            ax.plot(truth[index].detach().cpu().numpy(), label=f"{name} truth")
            for key, value in preds.items():
                ax.plot(value[index].detach().cpu().numpy(), label=key)
            if use_loglog:
                ax.set_xscale('log')
            ax.set_title(f"{name} sample {index}")
            ax.legend()
            ax.grid(True)
        for extra_ax in axes[len(indices):]:
            extra_ax.remove()
        plt.tight_layout()
        artifact_file = f"{name}_{self.base_artifact_file}" if len(self.curves_config) > 1 else self.base_artifact_file
        if hasattr(trainer.logger, "experiment"):
            exp = trainer.logger.experiment
            if hasattr(exp, "log_figure"):  # W&B
                exp.log_figure(trainer.logger.run_id, fig, artifact_file=artifact_file)
            elif hasattr(exp, "add_figure"):  # TensorBoard
                exp.add_figure("figure", fig)
            else:
                logger.warning("Logger does not support figure logging.")
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            plot_path = os.path.join(self.output_dir, artifact_file)
            plt.savefig(plot_path)
        plt.close(fig)
