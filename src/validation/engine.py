
"""Validation engine hierarchy for VAE and PairVAE models."""

from __future__ import annotations

import json
from multiprocessing import cpu_count
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.transformations import Pipeline
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE
from src.validation.metrics import BaseFitMetric, FitResult, LesFitMetric, SaxsFitMetric
from src.logging_utils import get_logger


logger = get_logger(__name__)


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors to a device."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


def _extract_reconstruction(outputs: Any) -> Optional[torch.Tensor]:
    """Retrieve reconstruction tensor from model outputs."""

    if isinstance(outputs, dict):
        recon = outputs.get("recon")
        if recon is None:
            for key, value in outputs.items():
                if "recon" in key:
                    return value
            for value in outputs.values():
                if isinstance(value, torch.Tensor):
                    return value
        return recon
    if isinstance(outputs, (tuple, list)):
        return outputs[1] if len(outputs) > 1 else outputs[0]
    if isinstance(outputs, torch.Tensor):
        return outputs
    return None


def _evaluate_reconstruction_sample(
    true_vals: np.ndarray, pred_vals: np.ndarray, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute reconstruction metrics for a single sample."""

    true_array = np.asarray(true_vals, dtype=float).reshape(-1)
    pred_array = np.asarray(pred_vals, dtype=float).reshape(-1)

    mae = mean_absolute_error(true_array, pred_array)
    mse = mean_squared_error(true_array, pred_array)
    r2 = r2_score(true_array, pred_array)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "r2_score": float(r2),
        "rmse": float(np.sqrt(mse)),
        "signal_length": int(len(pred_array)),
        **metadata,
    }

    return {"metrics": metrics, "true_vals": true_array, "pred_vals": pred_array}


class BaseValidationEngine(ABC):
    """Common validation workflow for reconstruction and fit metrics."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        data_path: str,
        output_dir: str,
        config: Dict[str, Any],
        conversion_dict_path: Optional[str] = None,
        batch_size: int = 32,
        eval_percentage: float = 0.1,
        fit_percentage: float = 0.0005,
        qmin_fit: float = 0.001,
        qmax_fit: float = 0.5,
        factor_scale_to_conc: float = 20878,
        n_processes: Optional[int] = None,
        random_state: int = 42,
        signal_length: Optional[int] = None,
        mode: Optional[str] = None
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.model_type = config.get("model_config", {}).get("type", "unknown")
        self.model_spec = config.get("model_config", {}).get("spec", "unknown")
        logger.info("Model type: %s", self.model_type)
        logger.info("Model spec: %s", self.model_spec)
        self.batch_size = batch_size
        self.eval_percentage = eval_percentage
        self.fit_percentage = fit_percentage
        self.qmin_fit = qmin_fit
        self.qmax_fit = qmax_fit
        self.factor_scale_to_conc = factor_scale_to_conc
        self.n_processes = n_processes or max(1, cpu_count() - 1)
        self.random_state = random_state
        self.signal_length = signal_length
        self.mode = mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model()
        self.model.to(self.device).eval()

        conversion_dict = self._load_conversion_dict(conversion_dict_path)
        self.dataset, self.invert_transforms = self._build_dataset(conversion_dict)

        np.random.seed(self.random_state)

    def _load_conversion_dict(self, conversion_dict_path: Optional[str]) -> Optional[dict]:
        """Load conversion dictionary from file or config."""

        if conversion_dict_path:
            with open(conversion_dict_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return self.config.get("conversion_dict", {})

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """Instantiate the Lightning module."""

    @abstractmethod
    def _build_dataset(self, conversion_dict: Optional[dict]) -> Tuple[HDF5Dataset, Any]:
        """Create dataset and inverse transform callable."""

    def supports_reconstruction(self) -> bool:
        """Return whether reconstruction metrics are applicable."""

        return True

    @abstractmethod
    def forward_reconstruction(
        self, batch: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any], Optional[torch.Tensor]]:
        """Return reconstruction outputs, ground truth, metadata, and lengths."""

    @abstractmethod
    def forward_fit(self, batch: Dict[str, Any]) -> Tuple[List[Tuple], List[Tuple]]:
        """Return prediction and ground-truth samples for fit metrics."""

    @abstractmethod
    def create_fit_metric(self) -> BaseFitMetric:
        """Instantiate fit metric implementation."""

    def _subset_loader(self, fraction: float) -> DataLoader:
        """Create data loader for a random subset of the dataset."""

        total_samples = len(self.dataset)
        if total_samples == 0:
            raise ValueError("Dataset is empty.")
        n_samples = max(1, int(total_samples * fraction))
        indices = np.random.choice(total_samples, n_samples, replace=False)
        subset = Subset(self.dataset, indices)
        logger.info(
            "[BaseValidationEngine] Using %d samples out of %d (%.2f%%)",
            n_samples,
            total_samples,
            fraction * 100,
        )
        return DataLoader(subset, batch_size=self.batch_size, shuffle=False)

    def compute_reconstruction_metrics(self) -> Dict[str, Any]:
        """Compute signal reconstruction quality metrics."""

        if not self.supports_reconstruction():
            return {
                "samples_evaluated": 0,
                "reconstruction_status": "reconstruction not supported",
                "samples_seen": 0,
                "batches_processed": 0,
                "points_evaluated": 0,
            }

        loader = self._subset_loader(self.eval_percentage)

        sample_entries: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
        total_batches = 0
        attempted_samples = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing reconstruction metrics"):
                total_batches += 1
                batch = move_to_device(batch, self.device)
                recon, true_values, metadata_batch, lengths = self.forward_reconstruction(batch)

                if recon is None or true_values is None:
                    continue

                if recon.ndim == 3:
                    recon = recon.squeeze(1)

                true_np = true_values.detach().cpu().numpy()
                recon_np = recon.detach().cpu().numpy()

                attempted_samples += recon_np.shape[0]

                for i in range(recon_np.shape[0]):
                    if self.signal_length is not None:
                        max_len = min(self.signal_length, true_np.shape[1], recon_np.shape[1])
                        true_vals = true_np[i][:max_len].flatten()
                        pred_vals = recon_np[i][:max_len].flatten()
                    elif lengths is not None:
                        length = int(lengths[i].item())
                        true_vals = true_np[i][:length].flatten()
                        pred_vals = recon_np[i][:length].flatten()
                    else:
                        true_vals = true_np[i].flatten()
                        pred_vals = recon_np[i].flatten()

                    sample_meta: Dict[str, Any] = {}
                    for key, values in metadata_batch.items():
                        if hasattr(values, "__getitem__") and len(values) > i:
                            val = values[i]
                            if hasattr(val, "item"):
                                val = val.item()
                            sample_meta[key] = val

                    sample_entries.append((true_vals, pred_vals, sample_meta))

        if not sample_entries:
            return {
                "samples_evaluated": 0,
                "reconstruction_status": "no reconstruction predictions available",
                "batches_processed": total_batches,
                "samples_seen": attempted_samples,
            }

        parallel_results = Parallel(n_jobs=self.n_processes)(
            delayed(_evaluate_reconstruction_sample)(true_vals, pred_vals, metadata)
            for true_vals, pred_vals, metadata in sample_entries
        )

        detailed_results: List[Dict[str, Any]] = []
        all_pred_arrays: List[np.ndarray] = []
        all_true_arrays: List[np.ndarray] = []
        total_points = 0

        for result in parallel_results:
            metrics_entry = result["metrics"]
            detailed_results.append(metrics_entry)
            pred_values = result["pred_vals"]
            true_values = result["true_vals"]
            total_points += len(pred_values)
            all_pred_arrays.append(pred_values)
            all_true_arrays.append(true_values)

        all_pred = np.concatenate(all_pred_arrays)
        all_true = np.concatenate(all_true_arrays)

        metrics = {
            "global_mae": float(mean_absolute_error(all_true, all_pred)),
            "global_mse": float(mean_squared_error(all_true, all_pred)),
            "global_rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
            "global_r2": float(r2_score(all_true, all_pred)),
            "samples_evaluated": len(detailed_results),
            "samples_processed": len(detailed_results),
            "samples_seen": attempted_samples,
            "batches_processed": total_batches,
            "points_evaluated": int(total_points),
            "reconstruction_status": "ok",
        }

        df = pd.DataFrame(detailed_results)
        csv_path = self.output_dir / "reconstruction_metrics_detailed.csv"
        df.to_csv(csv_path, index=False)

        metrics["reconstruction_details_path"] = str(csv_path)
        return metrics

    def compute_fit_metrics(self) -> Dict[str, Any]:
        """Compute geometric and concentration fit metrics."""

        loader = self._subset_loader(self.fit_percentage)
        metric = self.create_fit_metric()

        pred_samples: List[Tuple] = []
        true_samples: List[Tuple] = []
        for batch in tqdm(loader, desc="Preprocess for fit samples"):
            with torch.no_grad():
                batch = move_to_device(batch, self.device)
                pred_batch, true_batch = self.forward_fit(batch)
                pred_samples.extend(pred_batch)
                true_samples.extend(true_batch)

        if not pred_samples:
            raise ValueError("No prediction samples available for fit metrics.")
        
        logger.info("Collecting fit samples using %d processes...", self.n_processes)

        pred_results = Parallel(n_jobs=self.n_processes)(
            delayed(metric.safe_fit_single)(sample) for sample in pred_samples
        )
        if true_samples:
            true_results = (
                Parallel(n_jobs=self.n_processes)(
                    delayed(metric.safe_fit_single)(sample) for sample in true_samples
                )
                if true_samples
                else []
            )
        else:
            true_results = []
        if all(r is None for r in pred_results):
            raise ValueError("All prediction fit attempts failed.")

        return self._aggregate_fit_results(pred_results, true_results)

    def _aggregate_fit_results(
        self,
        pred_results: Sequence[FitResult],
        true_results: Sequence[FitResult],
    ) -> Dict[str, Any]:
        """Aggregate per-sample fit outputs into summary metrics."""

        pred_attempted = len(pred_results)
        true_attempted = len(true_results)

        pred_successful = [r for r in pred_results if r is not None]
        true_successful = [r for r in true_results if r is not None]

        results: Dict[str, Any] = {
            "fit_pred_samples": len(pred_successful),
            "fit_pred_attempted": pred_attempted,
            "fit_pred_failures": pred_attempted - len(pred_successful),
            "fit_true_samples": len(true_successful),
            "fit_true_attempted": true_attempted,
            "fit_true_failures": true_attempted - len(true_successful),
        }
        detailed_rows: List[Dict[str, Any]] = []

        if pred_successful:
            pred_diam_errors = [r[0] for r in pred_successful]
            pred_length_errors = [r[1] for r in pred_successful]
            pred_conc_errors = [r[2] for r in pred_successful]

            for i, (
                diam_err,
                length_err,
                conc_err,
                pred_diam,
                pred_length,
                pred_conc,
                true_diam,
                true_length,
                true_conc,
            ) in enumerate(pred_successful):
                detailed_rows.append(
                    {
                        "sample_index": i,
                        "type": "prediction",
                        "true_diameter_nm": true_diam,
                        "true_length_nm": true_length,
                        "true_concentration": true_conc,
                        "pred_diameter_nm": pred_diam,
                        "pred_length_nm": pred_length,
                        "pred_concentration": pred_conc,
                        "diameter_abs_error": diam_err,
                        "length_abs_error": length_err,
                        "concentration_abs_error": conc_err,
                    }
                )

            results.update(
                {
                    "fit_pred_samples": len(pred_successful),
                    "mae_diameter_nm_pred": float(np.mean(pred_diam_errors)),
                    "mae_length_nm_pred": float(np.mean(pred_length_errors)),
                    "mae_concentration_pred": float(np.mean(pred_conc_errors)),
                    "mse_diameter_nm_pred": float(np.mean(np.square(pred_diam_errors))),
                    "mse_length_nm_pred": float(np.mean(np.square(pred_length_errors))),
                    "mse_concentration_pred": float(np.mean(np.square(pred_conc_errors))),
                    "rmse_diameter_nm_pred": float(np.sqrt(np.mean(np.square(pred_diam_errors)))),
                    "rmse_length_nm_pred": float(np.sqrt(np.mean(np.square(pred_length_errors)))),
                    "rmse_concentration_pred": float(np.sqrt(np.mean(np.square(pred_conc_errors)))),
                }
            )

        if true_successful:
            true_diam_errors = [r[0] for r in true_successful]
            true_length_errors = [r[1] for r in true_successful]
            true_conc_errors = [r[2] for r in true_successful]

            for i, (
                diam_err,
                length_err,
                conc_err,
                pred_diam,
                pred_length,
                pred_conc,
                true_diam,
                true_length,
                true_conc,
            ) in enumerate(true_successful):
                detailed_rows.append(
                    {
                        "sample_index": i,
                        "type": "ground_truth",
                        "true_diameter_nm": true_diam,
                        "true_length_nm": true_length,
                        "true_concentration": true_conc,
                        "pred_diameter_nm": pred_diam,
                        "pred_length_nm": pred_length,
                        "pred_concentration": pred_conc,
                        "diameter_abs_error": diam_err,
                        "length_abs_error": length_err,
                        "concentration_abs_error": conc_err,
                    }
                )

            results.update(
                {
                    "fit_true_samples": len(true_successful),
                    "mae_diameter_nm_true": float(np.mean(true_diam_errors)),
                    "mae_length_nm_true": float(np.mean(true_length_errors)),
                    "mae_concentration_true": float(np.mean(true_conc_errors)),
                    "mse_diameter_nm_true": float(np.mean(np.square(true_diam_errors))),
                    "mse_length_nm_true": float(np.mean(np.square(true_length_errors))),
                    "mse_concentration_true": float(np.mean(np.square(true_conc_errors))),
                    "rmse_diameter_nm_true": float(np.sqrt(np.mean(np.square(true_diam_errors)))),
                    "rmse_length_nm_true": float(np.sqrt(np.mean(np.square(true_length_errors)))),
                    "rmse_concentration_true": float(np.sqrt(np.mean(np.square(true_conc_errors)))),
                }
            )

        if pred_successful and true_successful:
            results.update(
                {
                    "fit_diameter_mae_ratio": float(
                        results["mae_diameter_nm_pred"] / (results["mae_diameter_nm_true"] + 1e-9)
                    ),
                    "fit_length_mae_ratio": float(
                        results["mae_length_nm_pred"] / (results["mae_length_nm_true"] + 1e-9)
                    ),
                    "fit_concentration_mae_ratio": float(
                        results["mae_concentration_pred"] / (results["mae_concentration_true"] + 1e-9)
                    ),
                }
            )

        if detailed_rows:
            df = pd.DataFrame(detailed_rows)
            csv_path = self.output_dir / "fit_detailed_results.csv"
            df.to_csv(csv_path, index=False)
            results["fit_details_path"] = str(csv_path)

        results.update(
            {
                "fit_total_processed": pred_attempted,
                "fit_percentage": self.fit_percentage,
            }
        )

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Persist validation outputs to disk."""

        yaml_path = self.output_dir / "validation_metrics.yaml"
        summary_path = self.output_dir / "metrics_summary.txt"

        with open(yaml_path, "w", encoding="utf-8") as handle:
            yaml.dump(results, handle, allow_unicode=True)

        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("VALIDATION SUMMARY\n")
            handle.write("=" * 80 + "\n\n")
            handle.write(f"Model type: {results.get('model_type', 'N/A')}\n")
            if "mode" in results:
                handle.write(f"Mode: {results['mode']}\n")
            handle.write(f"Checkpoint: {results.get('checkpoint_path', 'N/A')}\n")
            handle.write(f"Dataset: {results.get('data_path', 'N/A')}\n")
            handle.write(f"Random state: {results.get('random_state', 'N/A')}\n\n")

            handle.write("RECONSTRUCTION\n")
            handle.write("-" * 80 + "\n")
            handle.write(
                f"Status: {results.get('reconstruction_status', 'N/A')} | "
                f"Samples: {results.get('samples_evaluated', 0)} | "
                f"Points: {results.get('points_evaluated', 0)}\n"
            )
            if "global_mae" in results:
                handle.write(
                    f"MAE={results['global_mae']:.6f} "
                    f"RMSE={results.get('global_rmse', 0.0):.6f} "
                    f"R2={results.get('global_r2', 0.0):.6f}\n\n"
                )
            else:
                handle.write("No reconstruction metrics available.\n\n")

            handle.write("FIT METRICS\n")
            handle.write("-" * 80 + "\n")
            handle.write(
                f"Predicted samples: {results.get('fit_pred_samples', 0)} / "
                f"{results.get('fit_pred_attempted', 0)} (failures: {results.get('fit_pred_failures', 0)})\n"
            )
            handle.write(
                f"Ground-truth samples: {results.get('fit_true_samples', 0)} / "
                f"{results.get('fit_true_attempted', 0)} (failures: {results.get('fit_true_failures', 0)})\n"
            )

            if "mae_diameter_nm_pred" in results:
                handle.write(
                    f"MAE diameter (pred): {results['mae_diameter_nm_pred']:.4f} | "
                    f"RMSE: {results['rmse_diameter_nm_pred']:.4f}\n"
                )
            if "mae_diameter_nm_true" in results:
                handle.write(
                    f"MAE diameter (true): {results['mae_diameter_nm_true']:.4f} | "
                    f"RMSE: {results['rmse_diameter_nm_true']:.4f}\n"
                )
            if "mae_length_nm_pred" in results:
                handle.write(
                    f"MAE length (pred): {results['mae_length_nm_pred']:.4f} | "
                    f"RMSE: {results['rmse_length_nm_pred']:.4f}\n"
                )
            if "mae_length_nm_true" in results:
                handle.write(
                    f"MAE length (true): {results['mae_length_nm_true']:.4f} | "
                    f"RMSE: {results['rmse_length_nm_true']:.4f}\n"
                )
            if "mae_concentration_pred" in results:
                handle.write(
                    f"MAE concentration (pred): {results['mae_concentration_pred']:.4e} | "
                    f"RMSE: {results['rmse_concentration_pred']:.4e}\n"
                )
            if "mae_concentration_true" in results:
                handle.write(
                    f"MAE concentration (true): {results['mae_concentration_true']:.4e} | "
                    f"RMSE: {results['rmse_concentration_true']:.4e}\n"
                )
            if "fit_diameter_mae_ratio" in results:
                handle.write(
                    f"Diameter MAE ratio (pred/true): {results['fit_diameter_mae_ratio']:.4f}\n"
                )
            if "fit_length_mae_ratio" in results:
                handle.write(
                    f"Length MAE ratio (pred/true): {results['fit_length_mae_ratio']:.4f}\n"
                )
            if "fit_concentration_mae_ratio" in results:
                handle.write(
                    f"Concentration MAE ratio (pred/true): {results['fit_concentration_mae_ratio']:.4f}\n"
                )

        results["yaml_path"] = str(yaml_path)
        results["summary_path"] = str(summary_path)

    def run(self) -> Dict[str, Any]:
        """Execute validation and save the aggregated metrics."""

        reconstruction_results = self.compute_reconstruction_metrics()
        fit_results = self.compute_fit_metrics()
        results = {**reconstruction_results, **fit_results}
        results.update(
            {
                "random_state": self.random_state,
                "model_type": self.model_type,
                "model_spec": self.model_spec,
                "checkpoint_path": str(self.checkpoint_path),
                "data_path": str(self.data_path),
            }
        )
        if self.mode is not None:
            results["mode"] = self.mode
        self._save_results(results)
        return results


class VaeValidationEngine(BaseValidationEngine):
    """Validation engine for standard VAE checkpoints."""

    def __init__(self, *, mode: Optional[str] = None, **kwargs: Any) -> None:
        if mode is not None:
            raise ValueError("Mode is not compatible with vanilla VAE models.")
        super().__init__(mode=None, **kwargs)

    def _load_model(self) -> torch.nn.Module:
        """Instantiate the VAE model."""

        return PlVAE.load_from_checkpoint(self.checkpoint_path)

    def _build_dataset(self, conversion_dict: Optional[dict]) -> Tuple[HDF5Dataset, Any]:
        """Configure dataset and inverse transforms for VAE."""

        dataset_cfg = self.config["global_config"]["dataset"]

        requested_metadata = [
            "length_nm",
            "diameter_nm",
            "concentration",
            "concentration_original",
        ]

        dataset = HDF5Dataset(
            str(self.data_path),
            transformer_q=Pipeline(self.model.get_transformer_q()),
            transformer_y=Pipeline(self.model.get_transformer_y()),
            metadata_filters=dataset_cfg["metadata_filters"],
            conversion_dict=conversion_dict,
            requested_metadata=requested_metadata,
            use_data_q=False,
        )
        invert_transforms = dataset.invert_transforms_func()
        return dataset, invert_transforms

    def forward_reconstruction(
        self, batch: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any], Optional[torch.Tensor]]:
        """Forward pass for reconstruction metrics."""

        outputs = self.model(batch)
        recon = _extract_reconstruction(outputs)
        true_values = batch.get("data_y")
        metadata = batch.get("metadata", {})
        lengths = batch.get("len")
        if true_values is not None:
            true_values = true_values.squeeze(1)
        return recon, true_values, metadata, lengths

    def forward_fit(self, batch: Dict[str, Any]) -> Tuple[List[Tuple], List[Tuple]]:
        """Prepare samples for VAE fit metrics."""

        outputs = self.model(batch)
        recon = _extract_reconstruction(outputs)
        if recon is None:
            raise ValueError("Model outputs must include reconstructions.")

        q_pred = None
        if isinstance(outputs, dict):
            q_pred = outputs.get("data_q")
        if q_pred is None:
            q_pred = batch.get("data_q")
        if q_pred is None:
            raise ValueError("Q values must be provided for fit metrics.")

        if recon.ndim == 3:
            recon = recon.squeeze(1)
        if q_pred.ndim == 3:
            q_pred = q_pred.squeeze(1)

        recon_inverted, _ = self.invert_transforms(
            recon.detach().cpu(), q_pred
        )

        true_values = batch["data_y"].detach().cpu().squeeze(1)
        true_inverted, _ = self.invert_transforms(
            true_values, q_pred
        )

        metadata = batch.get("metadata", {})

        diameter_key = "diameter_nm"
        length_key = "length_nm"
        concentration_key = None
        for key in ["concentration", "concentration_original"]:
            if key in metadata:
                concentration_key = key
                break
        if concentration_key is None or diameter_key not in metadata:
            raise ValueError("Metadata must include diameter and concentration values.")

        diam_true = metadata[diameter_key].detach().cpu().numpy()
        length_true = metadata[length_key].detach().cpu().numpy()
        conc_true = metadata[concentration_key].detach().cpu().numpy()

        pred_samples: List[Tuple] = []
        true_samples: List[Tuple] = []

        for i in range(recon_inverted.shape[0]):
            y_pred = recon_inverted[i].numpy()
            if isinstance(q_pred, list):
                q_slice = q_pred[i]
            else:
                q_slice = q_pred
            q_batch = q_slice.numpy() if hasattr(q_slice, "numpy") else q_slice
            true_d = float(diam_true[i]) if np.ndim(diam_true) > 0 else float(diam_true)
            true_l = float(length_true[i]) if np.ndim(length_true) > 0 else float(length_true)
            true_c = float(conc_true[i]) if np.ndim(conc_true) > 0 else float(conc_true)

            pred_samples.append((y_pred, q_batch, true_d, true_l, true_c))
            y_true = true_inverted[i].numpy()
            true_samples.append((y_true, q_batch, true_d, true_l, true_c))

        return pred_samples, true_samples

    def create_fit_metric(self) -> BaseFitMetric:
        """Create SAXS or LES fit metric for VAE."""
        if self.model_spec == 'les':
            return LesFitMetric(
            processes=self.n_processes,
        )
        elif self.model_spec == 'saxs':
            return SaxsFitMetric(
            processes=self.n_processes,
            qmin=self.qmin_fit,
            qmax=self.qmax_fit,
            scale_factor=self.factor_scale_to_conc,
            use_first_n_points=None,
        )
        else:
            raise ValueError(f"Unsupported model spec '{self.model_spec}' for VAE fit metric.")
        


class PairVaeValidationEngine(BaseValidationEngine):
    """Validation engine for PairVAE checkpoints."""

    def __init__(self, *, mode: Optional[str], **kwargs: Any) -> None:
        if mode is None:
            raise ValueError("Mode is required for PairVAE models.")
        if mode not in {"les_to_saxs", "saxs_to_saxs", "les_to_les", "saxs_to_les"}:
            raise ValueError(f"Unsupported mode {mode}.")
        super().__init__(mode=mode, **kwargs)

    def _load_model(self) -> torch.nn.Module:
        """Instantiate the PairVAE model."""

        return PlPairVAE.load_from_checkpoint(self.checkpoint_path)

    def _build_dataset(self, conversion_dict: Optional[dict]) -> Tuple[HDF5Dataset, Any]:
        """Configure dataset and inverse transforms for PairVAE."""

        dataset_cfg = self.config["global_config"]["dataset"]

        if self.mode == "les_to_saxs":
            transformer_q = Pipeline(self.model.get_transforms_data_les()["q"])
            transformer_y = Pipeline(self.model.get_transforms_data_les()['y'])
            output_transformer_q = Pipeline(self.model.get_transforms_data_saxs()["q"])
            output_transformer_y = Pipeline(self.model.get_transforms_data_saxs()['y'])
        elif self.mode == "les_to_les":
            transformer_q = Pipeline(self.model.get_transforms_data_les()["q"])
            transformer_y = Pipeline(self.model.get_transforms_data_les()['y'])
            output_transformer_q = Pipeline(self.model.get_transforms_data_les()["q"])
            output_transformer_y = Pipeline(self.model.get_transforms_data_les()['y'])
        elif self.mode == "saxs_to_saxs":
            transformer_q = Pipeline(self.model.get_transforms_data_saxs()["q"])
            transformer_y = Pipeline(self.model.get_transforms_data_saxs()['y'])
            output_transformer_q = Pipeline(self.model.get_transforms_data_saxs()["q"])
            output_transformer_y = Pipeline(self.model.get_transforms_data_saxs()['y'])
        else:
            transformer_q = Pipeline(self.model.get_transforms_data_saxs()["q"])
            transformer_y = Pipeline(self.model.get_transforms_data_saxs()['y'])
            output_transformer_q = Pipeline(self.model.get_transforms_data_les()["q"])
            output_transformer_y = Pipeline(self.model.get_transforms_data_les()['y'])

        requested_metadata = [
            "diameter_nm",
            "length_nm",
            "concentration_original",
            "concentration",
        ]

        dataset = HDF5Dataset(
            str(self.data_path),
            transformer_q=transformer_q,
            transformer_y=transformer_y,
            metadata_filters=dataset_cfg["metadata_filters"],
            conversion_dict=conversion_dict,
            requested_metadata=requested_metadata,
            use_data_q=False,
        )

        def invert(y, q):
            return output_transformer_y.invert(y), output_transformer_q.invert(q)

        return dataset, invert

    def supports_reconstruction(self) -> bool:
        """PairVAE reconstruction only available for matching modalities."""

        return self.mode in {"les_to_les", "saxs_to_saxs"}

    def forward_reconstruction(
        self, batch: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any], Optional[torch.Tensor]]:
        """Forward pass for reconstruction metrics."""

        if not self.supports_reconstruction():
            return None, None, {}, None

        if self.mode == "les_to_les":
            recon, _ = self.model.les_to_les(batch)
        else:
            recon, _ = self.model.saxs_to_saxs(batch)

        true_values = batch.get("data_y")
        metadata = batch.get("metadata", {})
        lengths = batch.get("len")
        if true_values is not None:
            true_values = true_values.squeeze(1)
        return recon, true_values, metadata, lengths

    def forward_fit(self, batch: Dict[str, Any]) -> Tuple[List[Tuple], List[Tuple]]:
        """Prepare samples for PairVAE fit metrics."""

        if self.mode == "les_to_saxs":
            recon, q_pred = self.model.les_to_saxs(batch)
            true_values = None
        elif self.mode == "saxs_to_saxs":
            recon, q_pred = self.model.saxs_to_saxs(batch)
            true_values = batch["data_y"].detach().cpu().squeeze(1)
        elif self.mode == "les_to_les":
            recon, q_pred = self.model.les_to_les(batch)
            true_values = batch["data_y"].detach().cpu().squeeze(1)
        else:
            recon, q_pred = self.model.saxs_to_les(batch)
            true_values = None

        if recon is None or q_pred is None:
            raise ValueError("Model outputs must provide reconstruction and q data.")

        if recon.ndim == 3:
            recon = recon.squeeze(1)
        if q_pred.ndim == 3:
            q_pred = q_pred.squeeze(1)

        recon_inverted, _ = self.invert_transforms(
            recon.detach().cpu(), q_pred
        )

        if true_values is not None:
            true_inverted, _ = self.invert_transforms(
                true_values, q_pred
            )
        else:
            true_inverted, _ = None, None

        metadata = batch.get("metadata", {})

        diameter_key = "diameter_nm"
        length_key = "length_nm"
        concentration_key = None
        for key in ["concentration", "concentration_original"]:
            if key in metadata:
                concentration_key = key
                break
        if concentration_key is None or diameter_key not in metadata:
            raise ValueError("Metadata must include diameter and concentration values.")

        diam_true = metadata[diameter_key].detach().cpu().numpy()
        length_true = metadata[length_key].detach().cpu().numpy()
        conc_true = metadata[concentration_key].detach().cpu().numpy()

        pred_samples: List[Tuple] = []
        true_samples: List[Tuple] = []

        for i in range(recon_inverted.shape[0]):
            y_pred = recon_inverted[i].numpy()
            q_batch = q_pred.numpy() if hasattr(q_pred, "numpy") else q_pred
            true_d = float(diam_true[i]) if np.ndim(diam_true) > 0 else float(diam_true)
            true_l = float(length_true[i]) if np.ndim(length_true) > 0 else float(length_true)
            true_c = float(conc_true[i]) if np.ndim(conc_true) > 0 else float(conc_true)

            pred_samples.append((y_pred, q_batch, true_d, true_l, true_c))

            if true_inverted is not None :
                y_true = true_inverted[i].numpy()
                true_samples.append((y_true, q_batch, true_d, true_l, true_c))

        return pred_samples, true_samples

    def create_fit_metric(self) -> BaseFitMetric:
        """Create LES or SAXS fit metric depending on mode."""
        assert self.model_type == 'pair_vae', "Model type must be 'pair' for PairVAE fit metric."
        if self.mode in {"les_to_saxs", "saxs_to_saxs"}:
            use_first = self.signal_length if self.mode == "les_to_saxs" else None
            return SaxsFitMetric(
                processes=self.n_processes,
                qmin=self.qmin_fit,
                qmax=self.qmax_fit,
                scale_factor=self.factor_scale_to_conc,
                use_first_n_points=use_first,
            )
        return LesFitMetric(
            processes=self.n_processes,
        )


def _load_config(checkpoint_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def ValidationEngine(**kwargs: Any) -> BaseValidationEngine:
    """Factory returning the appropriate validation engine implementation."""

    checkpoint_path = kwargs["checkpoint_path"]
    config = _load_config(checkpoint_path)
    model_type = config["model_config"]["type"].lower()

    if model_type == "vae":
        return VaeValidationEngine(config=config, **kwargs)
    if model_type == "pair_vae":
        return PairVaeValidationEngine(config=config, **kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")
