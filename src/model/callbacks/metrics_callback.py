"""Metrics callbacks used during Lightning training loops."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import Callback, LightningModule
from lmfit import Parameters, minimize
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel


TensorBatch = Any


def _first_dataloader(dataloaders: Any) -> Any:
    """Return the first dataloader from a Lightning ``dataloaders`` attribute."""

    if isinstance(dataloaders, (list, tuple)):
        return dataloaders[0]
    return dataloaders


def move_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """Recursively move tensors nested inside dicts, lists, or tuples to ``device``."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


class MAEMetricCallback(Callback):
    """Compute and log per-reconstruction mean absolute errors on validation batches."""

    def __init__(self) -> None:
        self.mae_loss = nn.L1Loss()
        self.best_mae: Dict[str, float] = {}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """Aggregate MAE metrics across the validation dataloader."""

        mae_dict: Dict[str, list[float]] = {}
        pl_module.eval()
        val_dataloader = _first_dataloader(trainer.val_dataloaders)
        with torch.no_grad():
            for batch in val_dataloader:
                batch = move_to_device(batch, pl_module.device)
                data_y = batch["data_y"]
                inputs = data_y.squeeze(1)
                outputs = pl_module(batch)
                if isinstance(outputs, dict):
                    items = outputs.items()
                elif isinstance(outputs, tuple):
                    items = [("recon", outputs[1])]
                else:
                    items = [("recon", outputs)]
                for key, value in items:
                    if "recon" not in key:
                        continue
                    recon = value
                    if isinstance(recon, torch.Tensor) and recon.ndim > inputs.ndim:
                        recon = recon.squeeze(1)
                    mae_value = self.mae_loss(recon, inputs).item()
                    mae_dict.setdefault(key, []).append(mae_value)
        metrics: Dict[str, float] = {}
        for key, values in mae_dict.items():
            avg_mae = float(sum(values) / len(values))
            best = self.best_mae.get(key, float("inf"))
            if avg_mae < best:
                self.best_mae[key] = avg_mae
            metrics[f"val_mae_{key}"] = avg_mae
            metrics[f"best_val_mae_{key}"] = self.best_mae[key]
        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)
        pl_module.train()