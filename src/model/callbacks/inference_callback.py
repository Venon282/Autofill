import os
from random import sample
from typing import Any, Dict, Iterable, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch

TensorBatch = Union[torch.Tensor, Dict[str, Any], Iterable[Any]]


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch


class InferencePlotCallback(pl.Callback):
    def __init__(
            self,
            curves_config: Dict[str, Dict[str, Any]],
            artifact_file: str = "plot.png",
            output_dir: str = None,
            num_samples: int = 10,
            every_n_epochs: int = 5,
    ) -> None:
        self.curves_config = curves_config
        self._check_config()
        self.base_artifact_file = artifact_file
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._infer_and_plot(trainer, pl_module)

    def _check_config(self) -> None:
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
                except:
                    raise ValueError(f"'pred_keys' for {name} must be a list")

    def _infer_and_plot(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        model.eval()
        dataloader = trainer.val_dataloaders
        device = model.device
        batch = next(iter(dataloader))
        batch = move_to_device(batch, device)
        with torch.inference_mode():
            raw_outputs = model(batch)
            for name, cfg in self.curves_config.items():
                truth = batch[cfg['truth_key']]
                truth = truth.squeeze(1) if isinstance(truth, torch.Tensor) and truth.ndim > 1 else truth
                preds = {}
                for key in cfg['pred_keys']:
                    val = raw_outputs[key]
                    preds[key] = val.squeeze(1) if isinstance(val, torch.Tensor) and val.ndim > 1 else val
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
        indices = sample(range(len(truth)), min(self.num_samples, len(truth)))
        fig, axs = plt.subplots(len(indices)//2, 2, figsize=(15, 10 * len(indices)//2))
        axs = axs.ravel()
        for row, i in enumerate(indices):
            ax = axs[row]
            plot_fn = ax.plot
            plot_fn(truth[i].cpu().numpy(), label=f"{name} truth")
            for (k, v) in preds.items():
                plot_fn(v[i].cpu().numpy(), label=k)
            if use_loglog:
                ax.set_xscale('log')
            ax.set_title(f"{name} truth vs recon {i}")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        if self.curves_config > 1:
            artifact_file = name + '_' + self.base_artifact_file
        else:
             artifact_file = self.base_artifact_file
        if hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log_figure(trainer.logger.run_id, fig, artifact_file=artifact_file)
        elif self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            plot_path = os.path.join(self.output_dir, artifact_file)
            plt.savefig(plot_path)
        else:
            print("Output directory not specified. Plot not saved.")
        plt.close()




