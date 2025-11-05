from __future__ import annotations

import math
import os
from pathlib import Path
from random import sample
from typing import Any, Dict, Iterable, Sequence, Union, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning.pytorch as pl

from src.logging_utils import get_logger

logger = get_logger(__name__)

"""Lightning callback utilities for running inference plots during validation."""

TensorBatch = Union[torch.Tensor, Dict[str, Any], Iterable[Any]]

def move_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """Recursively send nested tensors and collections to a device."""
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
    """Generate and log diagnostic plots from validation or training batches."""

    def __init__(
        self,
        curves_config: Dict[str, Dict[str, Any]],
        artifact_file: str = "plot.png",
        num_samples: int = 10,
        every_n_epochs: int = 5,
        plot_on_train: bool = True,
        plot_on_val: bool = True,
        output_dir: Optional[str, Path] = None,
    ) -> None:
        self.curves_config = curves_config
        self._validate_config()
        self.base_artifact_file = artifact_file
        self.output_dir = None
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.plot_on_train = plot_on_train
        self.plot_on_val = plot_on_val
        self.output_dir = output_dir
        if not (self.plot_on_train or self.plot_on_val):
            logger.warning("InferencePlotCallback initialized with both plot_on_train and plot_on_val set to False.")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run inference at validation end."""
        if self.plot_on_val and (trainer.current_epoch % self.every_n_epochs == 0):
            self._infer_and_plot(trainer, pl_module, mode="val")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Optionally run inference at training end."""
        if self.plot_on_train and (trainer.current_epoch % self.every_n_epochs == 0):
            self._infer_and_plot(trainer, pl_module, mode="train")

    def _validate_config(self) -> None:
        """Ensure the curve configuration dictionary is well-formed."""
        if not isinstance(self.curves_config, dict):
            raise ValueError("curves_config must be a dictionary")
        for name, cfg in self.curves_config.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Configuration for {name} must be a dictionary")
            if "truth_key" not in cfg or "pred_keys" not in cfg:
                raise ValueError(f"Configuration for {name} must contain 'truth_key' and 'pred_keys'")
            if not isinstance(cfg["pred_keys"], list):
                try:
                    cfg["pred_keys"] = list(cfg["pred_keys"])
                except TypeError as exc:
                    raise ValueError(f"'pred_keys' for {name} must be a list") from exc

    def _infer_and_plot(self, trainer: pl.Trainer, model: pl.LightningModule, mode: str) -> None:
        """Fetch a batch from train/val dataloader, run inference, and log plots."""
        model.eval()

        if mode == "train":
            dataloaders = trainer.train_dataloader
        else:
            dataloaders = trainer.val_dataloaders
        dataloader = dataloaders[0] if isinstance(dataloaders, (list, tuple)) else dataloaders
        batch = next(iter(dataloader))
        batch = move_to_device(batch, model.device)

        with torch.inference_mode():
            raw_outputs = model(batch)
            for name, cfg in self.curves_config.items():
                truth = batch[cfg["truth_key"]]
                if isinstance(truth, torch.Tensor) and truth.ndim > 1:
                    truth = truth.squeeze(1)
                preds = {
                    key: (raw_outputs[key].squeeze(1) if raw_outputs[key].ndim > 1 else raw_outputs[key])
                    for key in cfg["pred_keys"]
                }
                self._plot(trainer, name, truth, preds, cfg.get("use_loglog", False), f"{mode}_{self.base_artifact_file}")
        model.train()

    def _plot(
        self,
        trainer: pl.Trainer,
        name: str,
        truth: torch.Tensor,
        preds: Dict[str, torch.Tensor],
        use_loglog: bool,
        base_artifact_file: str = "plot.png",
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
                ax.set_xscale("log")
            ax.set_title(f"{name} sample {index}")
            ax.legend()
            ax.grid(True)

        for extra_ax in axes[len(indices):]:
            extra_ax.remove()

        plt.tight_layout()
        artifact_file = f"{name}_{base_artifact_file}" if len(self.curves_config) > 1 else base_artifact_file

        if hasattr(trainer.logger, "experiment"):
            exp = trainer.logger.experiment
            if hasattr(exp, "log_figure"):  # W&B
                exp.log_figure(trainer.logger.run_id, fig, artifact_file=artifact_file)
            elif hasattr(exp, "add_figure"):  # TensorBoard
                exp.add_figure(f"{name}/epoch_{trainer.current_epoch}", fig)
            else:
                logger.warning("Logger does not support figure logging.")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            root, ext = os.path.splitext(artifact_file)
            ext = ext or ".png"
            plot_path = os.path.join(self.output_dir, root + ext)
            fig.savefig(plot_path, bbox_inches="tight", dpi=200)
            # logger.info(f"Saved plot to {plot_path}")

        plt.close(fig)
