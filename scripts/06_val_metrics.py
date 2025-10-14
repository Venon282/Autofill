#!/usr/bin/env python3
"""
Optimized script to compute validation metrics on H5 dataset.
Refactored for better maintainability without metrics_callback dependency.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from joblib import Parallel, delayed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.transformations import Pipeline
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE

from dotenv import load_dotenv
load_dotenv()  # Charge les variables depuis .env

# PATH_TO_Q_SAXS = os.getenv("PATH_TO_Q_SAXS")
# PATH_TO_WL_LES = os.getenv("PATH_TO_WL_LES")

def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors to specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch

def lesfit_single_sample(args: Tuple) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Fit a cylindrical model on a LES sample to extract diameter and concentration.

    Args:
        args: Tuple containing (sample_data, qmin_fit, qmax_fit, factor_scale_to_conc, use_first_n_points)

    Returns:
        Tuple (diam_err, length_err, conc_err, pred_diam_nm, pred_length_nm, pred_conc, true_diam, true_length, true_conc) or None if failed
    """
    from scripts.utils.fit_cyl_Ag import fit_cyl_Ag

    sample_data = args
    y_np, wl, true_diam, true_length, true_conc = sample_data

    Longueur = 1E-3
    if wl is None : 
        wl = np.load(PATH_TO_WL_LES)

    results = fit_cyl_Ag(y_np,Longueur,wl)

    if results is not None :
        pred_diam_nm, pred_length_nm, concentration, _  = results

        diam_err = abs(pred_diam_nm - true_diam)
        length_err = abs(pred_length_nm - true_length)
        conc_err = abs(concentration - true_conc)

        return (diam_err, length_err, conc_err, pred_diam_nm, pred_length_nm, concentration, true_diam, true_length, true_conc)

    else : 

        return None

def sasfit_single_sample(args: Tuple) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Fit a cylindrical model on a SAXS sample to extract diameter and concentration.

    Args:
        args: Tuple containing (sample_data, qmin_fit, qmax_fit, factor_scale_to_conc, use_first_n_points)

    Returns:
        Tuple (diam_err, length_err, conc_err, pred_diam_nm, pred_length_nm, pred_conc, true_diam, true_length, true_conc) or None if failed
    """
    from lmfit import Parameters, minimize
    from sasmodels.core import load_model
    from sasmodels.data import empty_data1D
    from sasmodels.direct_model import DirectModel

    sample_data, qmin_fit, qmax_fit, factor, use_first_n_points = args
    y_np, q_np, true_diam, true_length, true_conc = sample_data

    # Select data points according to strategy
    if q_np is None and use_first_n_points is not None:
        # For les_to_saxs: use first N points instead of qmin/qmax filtering
        n_points = min(use_first_n_points, len(y_np))
        q_fit = q_np[:n_points]
        i_fit = y_np[:n_points]
    else:
        # For saxs_to_saxs: use qmin/qmax mask
        mask = (q_np >= qmin_fit) & (q_np <= qmax_fit)
        if not np.any(mask):
            raise ValueError(f"No data points in the specified q range. {qmin_fit} to {qmax_fit}, available q range: {q_np.min()} to {q_np.max()}")
        q_fit = q_np[mask]
        i_fit = y_np[mask]

    # Cylindrical model and physical parameters
    model = load_model("cylinder")
    data_fit = empty_data1D(q_fit)
    calc_fit = DirectModel(data_fit, model)
    sld_particle = 7.76211e11 * 1e-16  # Ag
    sld_solvent = 9.39845e10 * 1e-16   # H2O

    def residual_log(params):
        p = params.valuesdict()
        I = calc_fit(scale=p["scale"], background=p["background"],
                    radius=p["radius"], length=p["length"],
                    sld=sld_particle, sld_solvent=sld_solvent,
                    radius_pd=0.0, length_pd=0.0)
        eps = 1e-30
        return np.log10(np.clip(i_fit, eps, None)) - np.log10(np.clip(I, eps, None))

    # Parameter space
    params = Parameters()
    params.add("scale", value=1e14, min=1e6, max=1e20)
    params.add("radius", value=250.0, min=100.0, max=500.0)
    params.add("length", value=100.0, min=40.0, max=160.0)
    params.add("background", value=0, vary=False)

    # Deterministic DE
    res = minimize(
        residual_log, params,
        method="differential_evolution",
        minimizer_kws=dict(
            popsize=1000,
            maxiter=300,
            polish=True,
            tol=1e-6,
            updating="deferred",
            workers=1
        )
    )
    # Deterministic local polish
    res = minimize(
        residual_log, res.params,
        method="least_squares",
        loss="linear",
        f_scale=0.1,
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
        max_nfev=80000
    )

    # Extract results
    fitted_scale = res.params["scale"].value
    radius_A = res.params["radius"].value
    length_A = res.params["length"].value

    volume_A3 = np.pi * radius_A**2 * length_A
    volume_cm3 = volume_A3 * 1e-24

    concentration = fitted_scale / volume_cm3 / 1e12
    converted_scale = fitted_scale * factor
    radius_nm = np.rint(radius_A/10)
    pred_diam_nm = radius_nm*2
    pred_length_nm = np.rint(length_A/10)

    diam_err = abs(pred_diam_nm - true_diam)
    length_err = abs(pred_length_nm - true_length)
    conc_err = abs(concentration - true_conc)

    return (diam_err, length_err, conc_err, pred_diam_nm, pred_length_nm, concentration, true_diam, true_length, true_conc)


class ValidationMetricsCalculator:
    """Optimized validation metrics calculator."""

    def __init__(self, checkpoint_path: str, data_path: str, output_dir: str,
                 conversion_dict_path: Optional[str] = None, batch_size: int = 32,
                 eval_percentage: float = 0.1, fit_percentage: float = 0.0005,
                 qmin_fit: float = 0.001, qmax_fit: float = 0.3,
                 factor_scale_to_conc: float = 20878, n_processes: Optional[int] = None,
                 random_state: int = 42, signal_length: Optional[int] = None, mode=None):

        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.eval_percentage = eval_percentage
        self.fit_percentage = fit_percentage
        self.qmin_fit = qmin_fit
        self.qmax_fit = qmax_fit
        self.factor_scale_to_conc = factor_scale_to_conc
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.random_state = random_state
        self.signal_length = signal_length
        self.mode = mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model_and_config()
        self._setup_dataset(conversion_dict_path)

        print(f"✓ Model loaded: {self.model_type}")
        print(f"✓ Dataset: {len(self.dataset)} samples")
        print(f"✓ Device: {self.device}")
        if self.signal_length:
            print(f"✓ Forced signal length: {self.signal_length} points")

    def _load_model_and_config(self):
        """Load model and configuration from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.config = checkpoint['hyper_parameters']['config']
        self.model_type = self.config['model']['type'].lower()
        if self.model_type not in ['vae', 'pair_vae']:
            raise ValueError(f"Model type {self.model_type} is not supported for validation.")
        if self.model_type == 'pair_vae' and self.mode is None:
            raise ValueError("Please provide --mode for the PairVAE model.")
        if self.model_type == 'vae' and self.mode is not None:
            raise ValueError(f"Can't use --mode with model type vae")
        if self.model_type == 'vae':
            self.model = PlVAE.load_from_checkpoint(self.checkpoint_path)
        elif self.model_type == 'pair_vae':
            self.model = PlPairVAE.load_from_checkpoint(self.checkpoint_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model.to(self.device).eval()

    def _setup_dataset(self, conversion_dict_path: Optional[str]):
        """Configure H5 dataset with transformations for VAE/PairVAE."""
        if conversion_dict_path:
            with open(conversion_dict_path, 'r') as f:
                conversion_dict = json.load(f)
        else:
            conversion_dict = self.config.get("conversion_dict")

        if self.model_type == 'vae':
            if self.mode in ['les_to_saxs', 'saxs_to_saxs']:
                # VAE: SAXS dataset with SAXS transformations
                self.dataset = HDF5Dataset(
                    str(self.data_path),
                    sample_frac=1.0,
                    transformer_q=self.config["transforms_data"]["q"],
                    transformer_y=self.config["transforms_data"]["y"],
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=conversion_dict,
                    requested_metadata=['length_nm', 'diameter_nm', 'concentration_original'],
                    use_data_q=False
                )
            else:
                # VAE: LES dataset with LES transformations
                self.dataset = HDF5Dataset(
                    str(self.data_path),
                    sample_frac=1.0,
                    transformer_q=self.config["transforms_data"]["q"],
                    transformer_y=self.config["transforms_data"]["y"],
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=conversion_dict,
                    requested_metadata=['length_nm', 'diameter_nm', 'concentration'],
                    use_data_q=False
                )
            self.invert_transforms = self.dataset.invert_transforms_func()
        else:
            transform_config = self.config.get('transforms_data', {})
            
            if self.mode in ['les_to_saxs']:
                self.dataset = HDF5Dataset(
                    str(self.data_path),
                    sample_frac=1.0,
                    transformer_q=Pipeline(transform_config["q_les"]),
                    transformer_y=Pipeline(transform_config["y_les"]),
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=conversion_dict,
                    requested_metadata=['diameter_nm', 'length_nm', 'concentration_original', 'concentration'],
                    use_data_q=False
                )
                output_transformer_q = Pipeline(transform_config["q_saxs"])
                output_transformer_y = Pipeline(transform_config["y_saxs"])
                
            elif self.mode in ['les_to_les']:
                self.dataset = HDF5Dataset(
                    str(self.data_path),
                    sample_frac=1.0,
                    transformer_q=Pipeline(transform_config["q_les"]),
                    transformer_y=Pipeline(transform_config["y_les"]),
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=conversion_dict,
                    requested_metadata=['diameter_nm', 'length_nm', 'concentration_original', 'concentration'],
                    use_data_q=False 
                )
                output_transformer_q = Pipeline(transform_config["q_les"])
                output_transformer_y = Pipeline(transform_config["y_les"])
            
            elif self.mode in ['saxs_to_saxs']: 
                self.dataset = HDF5Dataset(
                    str(self.data_path),
                    sample_frac=1.0,
                    transformer_q=Pipeline(transform_config["q_saxs"]),
                    transformer_y=Pipeline(transform_config["y_saxs"]),
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=conversion_dict,
                    requested_metadata=['diameter_nm', 'length_nm', 'concentration_original', 'concentration'],
                    use_data_q=False 
                )
                output_transformer_q = Pipeline(transform_config["q_saxs"])
                output_transformer_y = Pipeline(transform_config["y_saxs"])

            elif self.mode in ['saxs_to_les']: 
                self.dataset = HDF5Dataset(
                    str(self.data_path),
                    sample_frac=1.0,
                    transformer_q=Pipeline(transform_config["q_saxs"]),
                    transformer_y=Pipeline(transform_config["y_saxs"]),
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=conversion_dict,
                    requested_metadata=['diameter_nm', 'length_nm', 'concentration_original', 'concentration'],
                    use_data_q=False 
                )
                output_transformer_q = Pipeline(transform_config["q_les"])
                output_transformer_y = Pipeline(transform_config["y_les"])

            # Inversion function for PairVAE
            def invert(y, q):
                y = output_transformer_y.invert(y)
                q = output_transformer_q.invert(q)
                return y, q
            self.invert_transforms = invert

    def _extract_reconstruction(self, outputs: Any) -> Optional[torch.Tensor]:
        """Extract reconstruction from model outputs."""
        if isinstance(outputs, dict):
            recon = outputs.get('recon')
            if recon is None:
                for key, value in outputs.items():
                    if "recon" in key:
                        return value
            return recon
        elif isinstance(outputs, (tuple, list)):
            return outputs[1] if len(outputs) > 1 else outputs[0]
        return outputs

    def compute_reconstruction_metrics(self) -> Dict[str, Any]:
        """Compute reconstruction metrics (MAE, MSE, R², RMSE)."""
        print(f"\n📊 Computing reconstruction metrics ({self.eval_percentage*100:.1f}% dataset)")

        total_samples = len(self.dataset)
        n_samples = int(total_samples * self.eval_percentage)
        indices = np.random.choice(total_samples, n_samples, replace=False)

        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        all_predictions, all_true_values = [], []
        detailed_results = []

        with torch.no_grad():
            # For PairVAE, only compute reconstruction metrics if input==output
            if self.model_type == 'pair_vae' and self.mode not in ('les_to_les', 'saxs_to_saxs'):
                print("Reconstruction metrics skipped for cross-domain PairVAE mode.")
                return {'samples_evaluated': 0}

            for batch in tqdm(loader, desc="Computing reconstruction metrics"):
                batch = move_to_device(batch, self.device)
                if self.model_type == 'vae':
                    outputs = self.model(batch)
                    recon = self._extract_reconstruction(outputs)
                    q_pred = outputs['data_q'] if 'data_q' in outputs else batch.get('data_q')
                else:
                    if self.mode == 'les_to_les':
                        recon, q_pred = self.model.les_to_les(batch)
                    elif self.mode == 'saxs_to_saxs':
                        recon, q_pred = self.model.saxs_to_saxs(batch)
                    else:
                        recon = None
                        
                if recon is None:
                    print("WARNING: No reconstruction available from the model outputs.")
                    continue

                true_values = batch["data_y"].squeeze(1)
                if recon.ndim == 3:
                    recon = recon.squeeze(1)
                # if true_values.ndim == 3:
                #     true_values = q_pred.squeeze(1)
                
                true_cpu = true_values.detach().cpu()
                recon_cpu = recon.detach().cpu()
                true_np = true_cpu.numpy()
                recon_np = recon_cpu.numpy()
                # Métadonnées
                metadata_batch = batch.get("metadata", {})
                lengths = batch.get("len")

                # Métriques par échantillon
                for i in range(recon_np.shape[0]):
                    # Troncature selon signal_length forcé ou longueur réelle
                    if self.signal_length is not None:
                        # Utiliser la taille de signal forcée
                        max_len = min(self.signal_length, true_np.shape[1], recon_np.shape[1])
                        true_vals = true_np[i][:max_len].flatten()
                        pred_vals = recon_np[i][:max_len].flatten()
                    elif lengths is not None:
                        # Utiliser la longueur réelle du dataset
                        length = int(lengths[i].item())
                        true_vals = true_np[i][:length].flatten()
                        pred_vals = recon_np[i][:length].flatten()
                    else:
                        # Utiliser la taille complète
                        true_vals = true_np[i].flatten()
                        pred_vals = recon_np[i].flatten()

                    mae = mean_absolute_error(true_vals, pred_vals)
                    mse = mean_squared_error(true_vals, pred_vals)
                    r2 = r2_score(true_vals, pred_vals)

                    # Métadonnées échantillon
                    sample_meta = {}
                    for key, values in metadata_batch.items():
                        if hasattr(values, '__getitem__') and len(values) > i:
                            val = values[i].item() if hasattr(values[i], 'item') else values[i]
                            sample_meta[key] = val

                    detailed_results.append({
                        'mae': mae, 'mse': mse, 'r2_score': r2,
                        'rmse': np.sqrt(mse), 'signal_length': len(pred_vals),
                        **sample_meta
                    })

                    all_predictions.extend(pred_vals)
                    all_true_values.extend(true_vals)

        # Métriques globales
        if all_predictions:
            all_pred = np.array(all_predictions)
            all_true = np.array(all_true_values)

            results = {
                'global_mae': float(mean_absolute_error(all_true, all_pred)),
                'global_mse': float(mean_squared_error(all_true, all_pred)),
                'global_rmse': float(np.sqrt(mean_squared_error(all_true, all_pred))),
                'global_r2': float(r2_score(all_true, all_pred)),
                'mean_prediction': float(np.mean(all_pred)),
                'mean_true': float(np.mean(all_true)),
                'std_prediction': float(np.std(all_pred)),
                'std_true': float(np.std(all_true)),
                'samples_evaluated': len(detailed_results),
                'total_data_points': len(all_predictions),
                'eval_percentage': self.eval_percentage
            }

            # Sauvegarde détails
            if detailed_results:
                df = pd.DataFrame(detailed_results)
                csv_path = self.output_dir / "reconstruction_metrics_detailed.csv"
                df.to_csv(csv_path, index=False)
                print(f"✓ Details saved: {csv_path}")

            print(f"  MAE: {results['global_mae']:.6f}")
            print(f"  MSE: {results['global_mse']:.6f}")
            print(f"  R²: {results['global_r2']:.6f}")

            return results

        return {'samples_evaluated': 0}

    def compute_fit_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques Fit (diamètre et concentration via fitting physique)."""
        # PairVAE: can only measure Fit if output is SAXS
        if self.model_type == 'pair_vae' and self.mode not in ('les_to_saxs', 'saxs_to_saxs', 'les_to_les', 'saxs_to_les'):
            raise ValueError("Fit metrics can only be computed for PairVAE in 'les_to_saxs', 'saxs_to_saxs', 'les_to_les', 'saxs_to_les' mode.")

        print(f"\n🔬 Computing Fit metrics ({self.fit_percentage*100:.2f}% dataset)")

        total_samples = len(self.dataset)
        n_samples = int(total_samples * self.fit_percentage)

        if n_samples == 0:
            return {'fit_samples': 0}

        indices = np.random.choice(total_samples, n_samples, replace=False)
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        pred_samples = [] 
        true_samples = []

        use_first_n_points = None
        if self.model_type == 'pair_vae' and self.mode == 'les_to_saxs':
            use_first_n_points = self.signal_length  
            print(f"  Mode les_to_saxs: using first {use_first_n_points} points instead of qmin/qmax filtering")
        else:
            print(f"  Mode {self.mode if self.model_type == 'pair_vae' else 'VAE'}: using qmin={self.qmin_fit}, qmax={self.qmax_fit}")

        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting Fit samples"):
                batch = move_to_device(batch, self.device)

                if self.model_type == 'vae':
                    outputs = self.model(batch)
                    recon = self._extract_reconstruction(outputs)
                    if recon is None:
                        continue
                    q_pred = outputs['data_q'] if 'data_q' in outputs else batch.get('data_q')
                    # Vérité terrain toujours disponible pour VAE
                    true_values = batch["data_y"].detach().cpu().squeeze(1)
                else:
                    # PairVAE: sortie SAXS requise
                    if self.mode == 'saxs_to_saxs':
                        recon, q_pred = self.model.saxs_to_saxs(batch)
                        true_values = batch["data_y"].detach().cpu().squeeze(1)
                    elif self.mode == 'les_to_saxs':
                        recon, q_pred = self.model.les_to_saxs(batch)
                        true_values = None
                    elif self.mode == 'les_to_les':
                        recon, q_pred = self.model.les_to_les(batch)
                        true_values = batch["data_y"].detach().cpu().squeeze(1)
                    elif self.mode == 'saxs_to_les':
                        recon, q_pred = self.model.saxs_to_les(batch)
                        true_values = None

                # Données prédites
                if recon.ndim == 3:
                    recon = recon.squeeze(1)
                if q_pred.ndim == 3:
                    q_pred = q_pred.squeeze(1)

                recon_inverted, _ = self.invert_transforms(recon.detach().cpu(), q_pred)
                # Q come from model input, no need to invert
                q_inverted = q_pred

                # Vérité terrain 
                if true_values is not None:
                    true_inverted, q_true_inverted = self.invert_transforms(true_values, q_pred)
                else:
                    true_inverted, q_true_inverted = None, None

                meta = batch.get("metadata", {})
                
                diameter_key = 'diameter_nm'
                length_key = 'length_nm'
                try:
                    concentration_key = next(k for k in ['concentration', 'concentration_original'] if k in meta)
                    concentration = meta[concentration_key]
                    diameter = meta[diameter_key]
                except (KeyError, StopIteration) as e:
                    raise ValueError(
                        f"Need '{diameter_key}' and either 'concentration' or 'concentration_original' "
                        f"in metadata: {meta} for Fit metrics."
                    ) from e

                diam_true = meta[diameter_key].detach().cpu().numpy()
                length_true = meta[length_key].detach().cpu().numpy()
                conc_true = meta[concentration_key].detach().cpu().numpy()

                for i in range(recon_inverted.shape[0]):
                    # Predicted data
                    y_pred = recon_inverted[i].numpy()
                    q_pred = q_inverted[i]

                    # True data (control)
                    if true_inverted is not None and q_true_inverted is not None:
                        y_true = true_inverted[i].numpy()
                        q_true = q_true_inverted[i].numpy()

                    t_d = float(diam_true[i]) if np.ndim(diam_true) > 0 else float(diam_true)
                    t_l = float(length_true[i]) if np.ndim(length_true) > 0 else float(length_true)
                    t_c = float(conc_true[i]) if np.ndim(conc_true) > 0 else float(conc_true)

                    pred_samples.append((y_pred, q_pred, t_d, t_l, t_c))
                    if true_inverted is not None and q_true_inverted is not None:
                        true_samples.append((y_true, q_true, t_d, t_l, t_c))

        if not pred_samples:
            raise ValueError("No pred_samples after collecting Fit samples")

        print(f"  Fitting {len(pred_samples)} samples on predictions... using {self.n_processes} cpus")
        if true_samples:
            print(f"  Fitting {len(true_samples)} samples on ground truth (control)... using {self.n_processes} cpus")

        if self.mode in ['les_to_saxs', 'saxs_to_saxs'] or self.model_type == 'vae':
            pred_fit_args = [(sample, self.qmin_fit, self.qmax_fit, self.factor_scale_to_conc, use_first_n_points)
                            for sample in pred_samples]
            true_fit_args = [(sample, self.qmin_fit, self.qmax_fit, self.factor_scale_to_conc, use_first_n_points)
                            for sample in true_samples] if true_samples else []

            pred_results = Parallel(n_jobs=self.n_processes)(delayed(sasfit_single_sample)(arg) for arg in pred_fit_args)
            true_results = Parallel(n_jobs=self.n_processes)(delayed(sasfit_single_sample)(arg) for arg in true_fit_args) if true_fit_args else []

        elif self.mode in ['saxs_to_les', 'les_to_les']:
            pred_fit_args = [(sample) for sample in pred_samples]
            true_fit_args = [(sample) for sample in true_samples] if true_samples else []

            pred_results = Parallel(n_jobs=self.n_processes)(delayed(lesfit_single_sample)(arg) for arg in pred_fit_args)
            true_results = Parallel(n_jobs=self.n_processes)(delayed(lesfit_single_sample)(arg) for arg in true_fit_args) if true_fit_args else []
            
        pred_successful = [r for r in pred_results if r is not None]
        true_successful = [r for r in true_results if r is not None]

        results_dict = {}

        if pred_successful:
            pred_diam_errors = [r[0] for r in pred_successful]
            pred_length_errors = [r[1] for r in pred_successful]
            pred_conc_errors = [r[2] for r in pred_successful]

            # Résultats détaillés prédictions
            pred_detailed = []
            for i, (diam_err, length_err, conc_err, pred_diam, pred_length, pred_conc, true_diam, true_length, true_conc) in enumerate(pred_successful):
                pred_detailed.append({
                    'sample_index': i,
                    'type': 'prediction',
                    'true_diameter_nm': true_diam,
                    'true_length_nm': true_length,
                    'true_concentration': true_conc,
                    'pred_diameter_nm': pred_diam,
                    'pred_length_nm': pred_length,
                    'pred_concentration': pred_conc,
                    'diameter_abs_error': diam_err,
                    'length_abs_error': length_err,
                    'concentration_abs_error': conc_err
                })

            results_dict.update({
                'fit_pred_samples': len(pred_successful),
                'mae_diameter_nm_pred': float(np.mean(pred_diam_errors)),
                'mae_length_nm_pred': float(np.mean(pred_length_errors)),
                'mae_concentration_pred': float(np.mean(pred_conc_errors)),
                'mse_diameter_nm_pred': float(np.mean([e**2 for e in pred_diam_errors])),
                'mse_length_nm_pred': float(np.mean([e**2 for e in pred_length_errors])),
                'mse_concentration_pred': float(np.mean([e**2 for e in pred_conc_errors])),
                'rmse_diameter_nm_pred': float(np.sqrt(np.mean([e**2 for e in pred_diam_errors]))),
                'rmse_length_nm_pred': float(np.sqrt(np.mean([e**2 for e in pred_length_errors]))),
                'rmse_concentration_pred': float(np.sqrt(np.mean([e**2 for e in pred_conc_errors])))
            })

        if true_successful:
            true_diam_errors = [r[0] for r in true_successful]
            true_length_errors = [r[1] for r in true_successful]
            true_conc_errors = [r[2] for r in true_successful]

            # Résultats détaillés vérité terrain
            true_detailed = []
            for i, (diam_err, length_err, conc_err, pred_diam, pred_length, pred_conc, true_diam, true_length, true_conc) in enumerate(true_successful):
                true_detailed.append({
                    'sample_index': i,
                    'type': 'prediction',
                    'true_diameter_nm': true_diam,
                    'true_length_nm': true_length,
                    'true_concentration': true_conc,
                    'pred_diameter_nm': pred_diam,
                    'pred_length_nm': pred_length,
                    'pred_concentration': pred_conc,
                    'diameter_abs_error': diam_err,
                    'length_abs_error': length_err,
                    'concentration_abs_error': conc_err
                })

            results_dict.update({
                'fit_true_samples': len(true_successful),
                'mae_diameter_nm_true': float(np.mean(true_diam_errors)),
                'mae_length_nm_true': float(np.mean(true_length_errors)),
                'mae_concentration_true': float(np.mean(true_conc_errors)),
                'mse_diameter_nm_true': float(np.mean([e**2 for e in true_diam_errors])),
                'mse_length_nm_true': float(np.mean([e**2 for e in true_length_errors])),
                'mse_concentration_true': float(np.mean([e**2 for e in true_conc_errors])),
                'rmse_diameter_nm_true': float(np.sqrt(np.mean([e**2 for e in true_diam_errors]))),
                'rmse_length_nm_true': float(np.sqrt(np.mean([e**2 for e in true_length_errors]))),
                'rmse_concentration_true': float(np.sqrt(np.mean([e**2 for e in true_conc_errors])))
            })

        if pred_successful or true_successful:
            print("\n" + "="*80)
            print("FIT METRICS SUMMARY")
            print("="*80)
            
            # En-tête du tableau
            header = f"{'Parameter':<15} {'Type':<12} {'MAE':<12} {'RMSE':<12} {'Min':<12} {'Median':<12} {'Max':<12}"
            print(header)
            print("-" * len(header))
            
            if pred_successful:
                print(f"{'Diameter (nm)':<15} {'Prediction':<12} {results_dict['mae_diameter_nm_pred']:<12.2f} "
                      f"{results_dict['rmse_diameter_nm_pred']:<12.2f} {np.min(pred_diam_errors):<12.2f} "
                      f"{np.median(pred_diam_errors):<12.2f} {np.max(pred_diam_errors):<12.2f}")
                
                print(f"{'Length (nm)':<15} {'Prediction':<12} {results_dict['mae_length_nm_pred']:<12.2f} "
                      f"{results_dict['rmse_length_nm_pred']:<12.2f} {np.min(pred_length_errors):<12.2f} "
                      f"{np.median(pred_length_errors):<12.2f} {np.max(pred_length_errors):<12.2f}")
                
                print(f"{'Concentration':<15} {'Prediction':<12} {results_dict['mae_concentration_pred']:<12.2e} "
                      f"{results_dict['rmse_concentration_pred']:<12.2e} {np.min(pred_conc_errors):<12.2e} "
                      f"{np.median(pred_conc_errors):<12.2e} {np.max(pred_conc_errors):<12.2e}")
            
            if true_successful:
                print(f"{'Diameter (nm)':<15} {'Ground Truth':<12} {results_dict['mae_diameter_nm_true']:<12.2f} "
                      f"{results_dict['rmse_diameter_nm_true']:<12.2f} {np.min(true_diam_errors):<12.2f} "
                      f"{np.median(true_diam_errors):<12.2f} {np.max(true_diam_errors):<12.2f}")
                
                print(f"{'Length (nm)':<15} {'Ground Truth':<12} {results_dict['mae_length_nm_true']:<12.2f} "
                      f"{results_dict['rmse_length_nm_true']:<12.2f} {np.min(true_length_errors):<12.2f} "
                      f"{np.median(true_length_errors):<12.2f} {np.max(true_length_errors):<12.2f}")
                
                print(f"{'Concentration':<15} {'Ground Truth':<12} {results_dict['mae_concentration_true']:<12.2e} "
                      f"{results_dict['rmse_concentration_true']:<12.2e} {np.min(true_conc_errors):<12.2e} "
                      f"{np.median(true_conc_errors):<12.2e} {np.max(true_conc_errors):<12.2e}")

        # Calcul et affichage des ratios de performance
        if pred_successful and true_successful:
            diam_ratio = results_dict['mae_diameter_nm_pred'] / (results_dict['mae_diameter_nm_true'] + 1e-9)
            length_ratio = results_dict['mae_length_nm_pred'] / (results_dict['mae_length_nm_true'] + 1e-9)
            conc_ratio = results_dict['mae_concentration_pred'] / (results_dict['mae_concentration_true'] + 1e-9)

            results_dict.update({
                'fit_diameter_mae_ratio': float(diam_ratio),
                'fit_length_mae_ratio': float(length_ratio),
                'fit_concentration_mae_ratio': float(conc_ratio)
            })
            
            print("\n" + "-"*50)
            print("           PERFORMANCE RATIOS (Pred/Truth)")
            print("-"*50)
            print(f"{'Parameter':<15} {'Ratio':<10}")
            print("-"*25)
            
            print(f"{'Diameter':<15} {diam_ratio:<10.2f}")
            print(f"{'Length':<15} {length_ratio:<10.2f}")
            print(f"{'Concentration':<15} {conc_ratio:<10.2f}")
            
            print("\nNote: Ratio = 1.0 means perfect match with ground truth")
            print("="*80)

        # Sauvegarde détails combinés
        if pred_successful or true_successful:
            all_detailed = []
            if pred_successful:
                all_detailed.extend(pred_detailed)
            if true_successful:
                all_detailed.extend(true_detailed)

            df = pd.DataFrame(all_detailed)
            csv_path = self.output_dir / "fit_detailed_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Fit details saved: {csv_path}")

        results_dict.update({
            'fit_total_processed': len(pred_samples),
            'fit_percentage': self.fit_percentage,
            'fit_transforms_inverted': True
        })

        return results_dict

    def run_validation(self) -> Dict[str, Any]:
        """Execute complete validation metrics calculation."""
        print(f"Starting validation - Random State: {self.random_state}")

        # Compute metrics
        reconstruction_metrics = self.compute_reconstruction_metrics()
        fit_metrics = self.compute_fit_metrics()

        # Merge results
        results = {**reconstruction_metrics, **fit_metrics}
        results.update({
            'random_state': self.random_state,
            'model_type': self.model_type,
            'checkpoint_path': str(self.checkpoint_path),
            'data_path': str(self.data_path)
        })

        # Save results
        self._save_results(results)

        print(f"\n✅ Validation completed - Results in {self.output_dir}")
        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save results as YAML and text with Fit table."""

        yaml_path = self.output_dir / "validation_metrics.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        summary_path = self.output_dir / "metrics_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=== VALIDATION METRICS ===\n\n")
            f.write(f"Model: {results.get('model_type', 'N/A')}\n")
            f.write(f"Reconstruction samples: {results.get('samples_evaluated', 0)}\n")
            f.write(f"Fit pred samples: {results.get('fit_pred_samples', 0)}\n")
            f.write(f"Fit true samples: {results.get('fit_true_samples', 0)}\n")
            f.write(f"Random state: {results.get('random_state', 'N/A')}\n\n")

            if 'global_mae' in results:
                f.write("RECONSTRUCTION:\n")
                f.write(f"  MAE: {results['global_mae']:.6f}\n")
                f.write(f"  RMSE: {results.get('global_rmse', 0.0):.6f}\n")
                f.write(f"  R²: {results.get('global_r2', 0.0):.6f}\n\n")

            if ('mae_diameter_nm_pred' in results) or ('mae_diameter_nm_true' in results):
                f.write("="*80 + "\n")
                f.write("FIT METRICS SUMMARY\n")
                f.write("="*80 + "\n")
                header = f"{'Parameter':<15} {'Type':<12} {'MAE':<12} {'RMSE':<12} {'Min':<12} {'Median':<12} {'Max':<12}"
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")

                if 'mae_diameter_nm_pred' in results:
                    f.write(f"{'Diameter (nm)':<15} {'Prediction':<12} {results['mae_diameter_nm_pred']:<12.2f} "
                            f"{results['rmse_diameter_nm_pred']:<12.2f} "
                            f"{results.get('min_diameter_nm_pred', 'N/A')!s:<12} "
                            f"{results.get('median_diameter_nm_pred', 'N/A')!s:<12} "
                            f"{results.get('max_diameter_nm_pred', 'N/A')!s:<12}\n")
                    f.write(f"{'Length (nm)':<15} {'Prediction':<12} {results['mae_length_nm_pred']:<12.2f} "
                            f"{results['rmse_length_nm_pred']:<12.2f} "
                            f"{results.get('min_length_nm_pred', 'N/A')!s:<12} "
                            f"{results.get('median_length_nm_pred', 'N/A')!s:<12} "
                            f"{results.get('max_length_nm_pred', 'N/A')!s:<12}\n")
                    f.write(f"{'Concentration':<15} {'Prediction':<12} {results['mae_concentration_pred']:<12.2e} "
                            f"{results['rmse_concentration_pred']:<12.2e} "
                            f"{results.get('min_concentration_pred', 'N/A')!s:<12} "
                            f"{results.get('median_concentration_pred', 'N/A')!s:<12} "
                            f"{results.get('max_concentration_pred', 'N/A')!s:<12}\n")

                if 'mae_diameter_nm_true' in results:
                    f.write(f"{'Diameter (nm)':<15} {'Ground Truth':<12} {results['mae_diameter_nm_true']:<12.2f} "
                            f"{results['rmse_diameter_nm_true']:<12.2f} "
                            f"{results.get('min_diameter_nm_true', 'N/A')!s:<12} "
                            f"{results.get('median_diameter_nm_true', 'N/A')!s:<12} "
                            f"{results.get('max_diameter_nm_true', 'N/A')!s:<12}\n")
                    f.write(f"{'Length (nm)':<15} {'Ground Truth':<12} {results['mae_length_nm_true']:<12.2f} "
                            f"{results['rmse_length_nm_true']:<12.2f} "
                            f"{results.get('min_length_nm_true', 'N/A')!s:<12} "
                            f"{results.get('median_length_nm_true', 'N/A')!s:<12} "
                            f"{results.get('max_length_nm_true', 'N/A')!s:<12}\n")
                    f.write(f"{'Concentration':<15} {'Ground Truth':<12} {results['mae_concentration_true']:<12.2e} "
                            f"{results['rmse_concentration_true']:<12.2e} "
                            f"{results.get('min_concentration_true', 'N/A')!s:<12} "
                            f"{results.get('median_concentration_true', 'N/A')!s:<12} "
                            f"{results.get('max_concentration_true', 'N/A')!s:<12}\n")

                if 'fit_diameter_mae_ratio' in results:
                    f.write("\n" + "-"*50 + "\n")
                    f.write("           PERFORMANCE RATIOS (Pred/Truth)\n")
                    f.write("-"*50 + "\n")
                    f.write(f"{'Parameter':<15} {'Ratio':<10}\n")
                    f.write("-"*25 + "\n")
                    f.write(f"{'Diameter':<15} {results['fit_diameter_mae_ratio']:<10.2f}\n")
                    f.write(f"{'Length':<15} {results['fit_length_mae_ratio']:<10.2f}\n")
                    f.write(f"{'Concentration':<15} {results['fit_concentration_mae_ratio']:<10.2f}\n")
                    f.write("\nNote: Ratio = 1.0 means perfect match with ground truth\n")
                    f.write("="*80 + "\n")

        print(f"✓ Results saved: YAML, text summary")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimized validation metrics calculation")

    parser.add_argument("-c", "--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("-d", "--data_path", required=True, help="HDF5 file path")
    parser.add_argument("-o", "--outputdir", required=True, help="Output directory")
    parser.add_argument('--mode', choices=['les_to_saxs', 'saxs_to_saxs', 'les_to_les','saxs_to_les'], required=False, default=None,
                        help='Conversion mode for PairVAE')
    parser.add_argument("--signal_length", type=int, help="Forced signal length", default=1000)
    parser.add_argument("-cd", "--conversion_dict", help="Metadata conversion file")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--eval_percentage", type=float, default=0.005,
                       help="% dataset for reconstruction metrics")
    parser.add_argument("--fit_percentage", type=float, default=0.005,
                       help="% dataset for Fit")

    parser.add_argument("--qmin_fit", type=float, default=0.001, help="Q min fitting")
    parser.add_argument("--qmax_fit", type=float, default=0.5, help="Q max fitting")
    parser.add_argument("--factor_scale_to_conc", type=float, default=20878,
                       help="Scale to concentration conversion factor")

    parser.add_argument("--n_processes", type=int, help="Number of Fit processes")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.data_path}")
    print(f"Output: {args.outputdir}")

    calculator = ValidationMetricsCalculator(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.outputdir,
        conversion_dict_path=args.conversion_dict,
        batch_size=args.batch_size,
        eval_percentage=args.eval_percentage,
        fit_percentage=args.fit_percentage,
        qmin_fit=args.qmin_fit,
        qmax_fit=args.qmax_fit,
        factor_scale_to_conc=args.factor_scale_to_conc,
        n_processes=args.n_processes,
        random_state=args.random_state,
        signal_length=args.signal_length,
        mode=args.mode
    )

    calculator.run_validation()


if __name__ == "__main__":
    main()