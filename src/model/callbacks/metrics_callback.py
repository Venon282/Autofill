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


def fit_single_sample(args: Tuple[Tuple[np.ndarray, np.ndarray, float, float], float, float, float]):
    """Fit a single SAXS sample using the cylinder model."""

    sample_data, qmin_fit, qmax_fit, factor = args
    y_np, q_np, target_diam, target_conc = sample_data

    mask = (q_np >= qmin_fit) & (q_np <= qmax_fit)
    if not np.any(mask):
        return None

    q_fit = q_np[mask]
    i_fit = y_np[mask]

    cyl_model = load_model("cylinder")
    sld_particle = 7.76211e11 * 1e-16
    sld_solvent = 9.39845e10 * 1e-16

    data_fit = empty_data1D(q_fit)
    calc_fit = DirectModel(data_fit, cyl_model)

    params = Parameters()
    params.add("scale", value=1e14 / factor, min=1e6 / factor, max=1e20 / factor)
    params.add("radius", value=250.0, min=90.0, max=510.0)
    params.add("length", value=100.0, min=40.0, max=160.0)
    params.add("background", value=0, vary=False)

    def residual_log(p):
        values = p.valuesdict()
        intensity = calc_fit(
            scale=values["scale"],
            background=values["background"],
            radius=values["radius"],
            length=values["length"],
            sld=sld_particle,
            sld_solvent=sld_solvent,
            radius_pd=0.0,
            length_pd=0.0,
        )
        eps = 1e-30
        return np.log10(np.clip(i_fit, eps, None)) - np.log10(np.clip(intensity, eps, None))

    try:
        res = minimize(
            residual_log,
            params,
            method="differential_evolution",
            minimizer_kws=dict(popsize=500, maxiter=150, polish=True, tol=1e-5, updating="deferred", workers=1),
        )
        res = minimize(
            residual_log,
            res.params,
            method="least_squares",
            loss="linear",
            f_scale=0.1,
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            max_nfev=40000,
        )
    except Exception:  # pragma: no cover - propagate failure silently to skip sample
        return None

    fitted_scale = res.params["scale"].value
    radius_angstrom = res.params["radius"].value
    predicted_conc = float(fitted_scale * factor)
    predicted_radius_nm = float(np.rint(radius_angstrom / 10.0))
    predicted_diam_nm = float(predicted_radius_nm * 2.0)

    diam_err = abs(predicted_diam_nm - target_diam)
    conc_err = abs(predicted_conc - target_conc)
    return (diam_err, conc_err, predicted_diam_nm, predicted_conc, target_diam, target_conc)


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


class SASFitMetricCallback(Callback):
    """Fit a cylinder model to reconstructed spectra and report MAE metrics."""

    def __init__(
        self,
        qmin_fit: float = 0.001,
        qmax_fit: float = 0.3,
        val_percentage: float = 0.1,
        factor_scale_to_conc: float = 20878,
        n_processes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.qmin_fit = qmin_fit
        self.qmax_fit = qmax_fit
        self.val_percentage = val_percentage
        self.factor = factor_scale_to_conc
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.best = {"diameter_mae": float("inf"), "concentration_mae": float("inf")}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """Sample validation predictions and fit the SAS model via multiprocessing."""

        pl_module.eval()
        val_loader = _first_dataloader(trainer.val_dataloaders)
        all_samples = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, pl_module.device)
                outputs = pl_module(batch)
                if isinstance(outputs, dict):
                    recon = outputs.get('recon')
                elif isinstance(outputs, tuple):
                    recon = outputs[1]
                else:
                    recon = outputs
                if recon is None:
                    continue
                y_pred = recon.detach().cpu()
                if y_pred.ndim == 3 and y_pred.size(1) == 1:
                    y_pred = y_pred.squeeze(1)
                q_batch = batch["data_q"].detach().cpu()
                if q_batch.ndim == 3 and q_batch.size(1) == 1:
                    q_batch = q_batch.squeeze(1)
                metadata = batch.get("metadata", {})
                if not (isinstance(metadata, dict) and 'diameter_nm' in metadata and 'concentration_original' in metadata):
                    continue
                diam_true = metadata['diameter_nm'].detach().cpu().numpy()
                conc_true = metadata['concentration_original'].detach().cpu().numpy()
                batch_size = y_pred.shape[0]
                for i in range(batch_size):
                    y_np = y_pred[i].numpy()
                    q_np = q_batch[i].numpy()
                    diam_val = float(diam_true[i]) if np.ndim(diam_true) > 0 else float(diam_true)
                    conc_val = float(conc_true[i]) if np.ndim(conc_true) > 0 else float(conc_true)
                    all_samples.append((y_np, q_np, diam_val, conc_val))

        total_samples = len(all_samples)
        n_samples = int(total_samples * self.val_percentage)
        if n_samples == 0:
            trainer.logger.log_metrics({"sasfit_samples": 0}, step=trainer.current_epoch)
            pl_module.train()
            return

        selected_indices = np.random.choice(total_samples, size=n_samples, replace=False)
        selected_samples = [all_samples[i] for i in selected_indices]
        fit_args = [(sample, self.qmin_fit, self.qmax_fit, self.factor) for sample in selected_samples]

        diam_abs_err = []
        conc_abs_err = []
        with mp.Pool(processes=self.n_processes) as pool:
            results = pool.map(fit_single_sample, fit_args)
        for result in results:
            if result is None:
                continue
            diam_err, conc_err, *_ = result
            diam_abs_err.append(diam_err)
            conc_abs_err.append(conc_err)

        metrics = {"sasfit_samples": len(diam_abs_err), "sasfit_total_available": total_samples,
                   "sasfit_percentage_used": self.val_percentage}
        if diam_abs_err and conc_abs_err:
            diam_mae = float(sum(diam_abs_err) / len(diam_abs_err))
            conc_mae = float(sum(conc_abs_err) / len(conc_abs_err))
            if diam_mae < self.best["diameter_mae"]:
                self.best["diameter_mae"] = diam_mae
            if conc_mae < self.best["concentration_mae"]:
                self.best["concentration_mae"] = conc_mae
            metrics.update({
                "val_mae_diameter_nm": diam_mae,
                "best_val_mae_diameter_nm": self.best["diameter_mae"],
                "val_mae_concentration": conc_mae,
                "best_val_mae_concentration": self.best["concentration_mae"],
            })
        trainer.logger.log_metrics(metrics, step=trainer.current_epoch)
        pl_module.train()
