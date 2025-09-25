#!/usr/bin/env python3
"""
Script optimisé pour calculer les métriques de validation sur un dataset H5.
Refactorisé pour être plus maintenable et sans dépendance à metrics_callback.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from joblib import Parallel, delayed
# Ajouter le chemin src au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset.datasetH5 import HDF5Dataset
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Déplace récursivement les tenseurs vers le device spécifié."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


def sasfit_single_sample(args: Tuple) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Fit un modèle cylindrique sur un échantillon SAXS pour récupérer diamètre et concentration.

    Args:
        args: Tuple contenant (sample_data, qmin_fit, qmax_fit, factor_scale_to_conc)

    Returns:
        Tuple (diam_err, conc_err, pred_diam_nm, pred_conc, true_diam, true_conc) ou None si échec
"""
    from lmfit import Parameters, minimize
    from sasmodels.core import load_model
    from sasmodels.data import empty_data1D
    from sasmodels.direct_model import DirectModel

    sample_data, qmin_fit, qmax_fit, factor = args
    y_np, q_np, true_diam, true_length, true_conc = sample_data

    # Masque pour la plage de q
    mask = (q_np >= qmin_fit) & (q_np <= qmax_fit)
    if not np.any(mask):
        return None

    q_fit = q_np[mask]
    i_fit = y_np[mask]

    # Modèle cylindrique et paramètres physiques
    model     = load_model("cylinder")
    data_fit  = empty_data1D(q_fit)
    calc_fit  = DirectModel(data_fit,  model)
    sld_particle = 7.76211e11 * 1e-16  # Ag
    sld_solvent  = 9.39845e10 * 1e-16  # H2O

    def residual_log(params):
        p = params.valuesdict()
        I = calc_fit(scale=p["scale"], background=p["background"],
                    radius=p["radius"], length=p["length"],
                    sld=sld_particle, sld_solvent=sld_solvent,
                    radius_pd=0.0, length_pd=0.0)
        eps = 1e-30
        return np.log10(np.clip(i_fit, eps, None)) - np.log10(np.clip(I, eps, None))

    # Espace de paramètres
    params = Parameters()
    # Dans sasmodels c'est scale mais c'est la concentration et je rajoute un facteur pour la remettre en cm-3
    params.add("scale", value=1e14/20878, min=1e6/20878, max=1e20/20878)
    # Use metadata: Rayon minimum 200 A-1 (20 nm) et 1000 A-1 (100 nm)
    #length : Min 50 A-1 (50 nm) et 150 A-1 (150 nm)
    params.add("radius", value=250.0, min=90.0, max=510.0)   # tighten a lot
    params.add("length", value=100.0, min=40.0,  max=160.0)

    # Pas de background dans notre cas
    bg_max = 0
    params.add("background", value=0,vary=False)

    # Deterministic DE
    res = minimize(
        residual_log, params,
        method="differential_evolution",
        minimizer_kws=dict(
            # seed=42,          # Seed
            popsize=1000,        # smaller = faster
            maxiter=300,       # moderate
            polish=True,
            tol=1e-6,
            updating="deferred",
            workers=1          # full determinism
        )
    )
    #Deterministic local polish
    res = minimize(
        residual_log, res.params,
        method="least_squares",
        loss="linear",    # fastest;"soft_l1"
        f_scale=0.1,
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
        max_nfev=80000
    )

    # Extraction des résultats
    #Récupération des données du fits
    fitted_scale = res.params["scale"].value
    radius_A = res.params["radius"].value
    length_A =res.params["length"].value

    converted_scale = fitted_scale * factor # concentration_original
    radius_nm = np.rint(radius_A/10)
    pred_diam_nm = radius_nm*2
    pred_length_nm = np.rint(length_A/10)

    diam_err = abs(pred_diam_nm - true_diam)
    length_err = abs(pred_length_nm - true_length)
    conc_err = abs(converted_scale - true_conc)

    # Evaluate best fit on full q and on fit q 
    best = res.params.valuesdict()
    I_fit  = calc_fit(scale=best["scale"], background=best["background"],
                    radius=best["radius"], length=best["length"],
                    sld=sld_particle, sld_solvent=sld_solvent,
                    radius_pd=0.0, length_pd=0.0)

    mae_pct = mean_absolute_error(I_fit, i_fit)

    return (mae_pct, diam_err, length_err, conc_err, pred_diam_nm, pred_length_nm, converted_scale, true_diam, true_length, true_conc)




class ValidationMetricsCalculator:
    """Calculateur optimisé des métriques de validation."""

    def __init__(self, checkpoint_path: str, data_path: str, output_dir: str,
                 conversion_dict_path: Optional[str] = None, batch_size: int = 32,
                 eval_percentage: float = 0.1, sasfit_percentage: float = 0.0005,
                 qmin_fit: float = 0.001, qmax_fit: float = 0.3,
                 factor_scale_to_conc: float = 20878, n_processes: Optional[int] = None,
                 random_state: int = 42, signal_length: Optional[int] = None):

        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.eval_percentage = eval_percentage
        self.sasfit_percentage = sasfit_percentage
        self.qmin_fit = qmin_fit
        self.qmax_fit = qmax_fit
        self.factor_scale_to_conc = factor_scale_to_conc
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.random_state = random_state
        self.signal_length = signal_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Charger config et modèle
        self._load_model_and_config()
        self._setup_dataset(conversion_dict_path)

        print(f"✓ Modèle chargé: {self.model_type}")
        print(f"✓ Dataset: {len(self.dataset)} échantillons")
        print(f"✓ Device: {self.device}")
        if self.signal_length:
            print(f"✓ Taille signal forcée: {self.signal_length} points")

    def _load_model_and_config(self):
        """Charge le modèle et la configuration depuis le checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.config = checkpoint['hyper_parameters']['config']
        self.model_type = self.config['model']['type'].lower()

        if self.model_type == 'vae':
            self.model = PlVAE.load_from_checkpoint(self.checkpoint_path)
        elif self.model_type == 'pair_vae':
            self.model = PlPairVAE.load_from_checkpoint(self.checkpoint_path)
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")

        self.model.to(self.device).eval()

    def _setup_dataset(self, conversion_dict_path: Optional[str]):
        """Configure le dataset H5 avec les transformations."""
        if conversion_dict_path:
            with open(conversion_dict_path, 'r') as f:
                conversion_dict = json.load(f)
        else:
            conversion_dict = self.config.get("conversion_dict")

        print(self.config["transforms_data"]["y"])

        self.dataset = HDF5Dataset(
            str(self.data_path),
            sample_frac=1.0,
            transformer_q=self.config["transforms_data"]["q"],
            transformer_y=self.config["transforms_data"]["y"],
            metadata_filters=self.config["dataset"]["metadata_filters"],
            conversion_dict=conversion_dict,
            requested_metadata=['diameter_nm', 'length_nm', 'concentration_original', 'concentration_scaled']
        )
        self.invert_transforms = self.dataset.invert_transforms_func()

    def _extract_reconstruction(self, outputs: Any) -> Optional[torch.Tensor]:
        """Extrait la reconstruction des outputs du modèle."""
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
        """Calcule les métriques de reconstruction (MAE, MSE, R², RMSE)."""
        print(f"\n📊 Calcul métriques reconstruction ({self.eval_percentage*100:.1f}% dataset)")

        # Échantillonnage
        # np.random.seed(self.random_state)
        # torch.manual_seed(self.random_state)

        total_samples = len(self.dataset)
        n_samples = int(total_samples * self.eval_percentage)
        indices = np.random.choice(total_samples, n_samples, replace=False)

        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        all_predictions, all_true_values = [], []
        detailed_results = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing reconstruction metrics"):
                batch = move_to_device(batch, self.device)

                # Prédiction
                outputs = self.model(batch)
                recon = self._extract_reconstruction(outputs)
                if recon is None:
                    continue

                # Préparation des données
                true_values = batch["data_y"].squeeze(1)

                if recon.ndim > true_values.ndim:
                    recon = recon.squeeze(1)

                # Inversion des transformations vers l'espace physique
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

                    # Calcul métriques
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
                print(f"✓ Détails sauvés: {csv_path}")

            print(f"  MAE: {results['global_mae']:.6f}")
            print(f"  R²: {results['global_r2']:.6f}")

            return results

        return {'samples_evaluated': 0}

    def compute_sasfit_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques SASFit (diamètre et concentration via fitting physique)."""
        print(f"\n🔬 Calcul métriques SASFit ({self.sasfit_percentage*100:.2f}% dataset)")

        # Échantillonnage
        # np.random.seed(self.random_state)
        total_samples = len(self.dataset)
        n_samples = int(total_samples * self.sasfit_percentage)

        if n_samples == 0:
            return {'sasfit_samples': 0}

        indices = np.random.choice(total_samples, n_samples, replace=False)
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        # Collecte des échantillons pour prédictions ET vérité terrain
        pred_samples = []  # Échantillons prédits
        true_samples = []  # Échantillons de vérité terrain

        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting SASFit samples"):
                batch = move_to_device(batch, self.device)

                outputs = self.model(batch)
                recon = self._extract_reconstruction(outputs)
                if recon is None:
                    continue

                # Données prédites
                q_batch = batch["data_q"].detach().cpu()
                if recon.ndim == 3:
                    recon = recon.squeeze(1)
                if q_batch.ndim == 3:
                    q_batch = q_batch.squeeze(1)

                recon_inverted, q_inverted = self.invert_transforms(recon.cpu(), q_batch)

                #temoin
                true_values = batch["data_y"].detach().cpu().squeeze(1)
                true_inverted, q_true_inverted = self.invert_transforms(true_values, q_batch)

                print(mean_absolute_error(batch["untransformed_data_y"], true_inverted))

                true_inverted = batch["untransformed_data_y"]

                meta = batch.get("metadata", {})
                if not ('diameter_nm' in meta and 'concentration_original' in meta):
                    continue

                diam_true = meta['diameter_nm'].detach().cpu().numpy()
                length_true = meta['length_nm'].detach().cpu().numpy()
                conc_true = meta['concentration_original'].detach().cpu().numpy()
                lengths = batch.get("len")

                # Collecte par échantillon
                for i in range(recon_inverted.shape[0]):
                    # Données prédites
                    y_pred = recon_inverted[i].numpy()
                    q_pred = q_inverted[i].numpy()

                    # Données vraies (témoin)
                    y_true = true_inverted[i].numpy()
                    q_true = q_true_inverted[i].numpy()

                    t_d = float(diam_true[i]) if np.ndim(diam_true) > 0 else float(diam_true)
                    t_l = float(length_true[i]) if np.ndim(length_true) > 0 else float(length_true)
                    t_c = float(conc_true[i]) if np.ndim(conc_true) > 0 else float(conc_true)

                    pred_samples.append((y_pred, q_pred, t_d, t_l, t_c))
                    true_samples.append((y_true, q_true, t_d, t_l, t_c))

        if not pred_samples or not true_samples:
            return {'sasfit_samples': 0}

        print(f"  Fitting {len(pred_samples)} échantillons sur prédictions...")
        print(f"  Fitting {len(true_samples)} échantillons sur vérité terrain (témoin)...")

        # Fitting parallèle pour les prédictions
        pred_fit_args = [(sample, self.qmin_fit, self.qmax_fit, self.factor_scale_to_conc)
                        for sample in pred_samples]

        # Fitting parallèle pour la vérité terrain
        true_fit_args = [(sample, self.qmin_fit, self.qmax_fit, self.factor_scale_to_conc)
                        for sample in true_samples]

        pred_results = Parallel(n_jobs=self.n_processes)(
            delayed(sasfit_single_sample)(arg) for arg in pred_fit_args
        )
        true_results = Parallel(n_jobs=self.n_processes)(
            delayed(sasfit_single_sample)(arg) for arg in true_fit_args
        )

        # Traitement résultats prédictions
        pred_successful = [r for r in pred_results if r is not None]
        true_successful = [r for r in true_results if r is not None]

        results_dict = {}

        if pred_successful:
            pred_mae_cb_fit = [r[0] for r in pred_successful]
            pred_diam_errors = [r[1] for r in pred_successful]
            pred_length_errors = [r[2] for r in pred_successful]
            pred_conc_errors = [r[3] for r in pred_successful]

            # Résultats détaillés prédictions
            pred_detailed = []
            for i, (mae_cb, diam_err, length_err, conc_err, pred_diam, pred_length, pred_conc, true_diam, true_length, true_conc) in enumerate(pred_successful):
                pred_detailed.append({
                    'sample_index': i,
                    'type': 'prediction',
                    'true_diameter_nm': true_diam,
                    'true_length_nm': true_length,
                    'true_concentration': true_conc,
                    'pred_diameter_nm': pred_diam,
                    'pred_length_nm': pred_length,
                    'pred_concentration': pred_conc,
                    'mae_cb': mae_cb,
                    'diameter_abs_error': diam_err,
                    'length_abs_error': length_err,
                    'concentration_abs_error': conc_err
                })

            results_dict.update({
                'sasfit_pred_samples': len(pred_successful),
                'mae_cb_pred': float(np.mean(pred_mae_cb_fit)),
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

            print(f"  📊 Prédictions - MAE entre courbes fittées: {results_dict['mae_cb_pred']:.2f}")
            print(f"  📊 Prédictions - Diamètre MAE: {results_dict['mae_diameter_nm_pred']:.2f} nm")
            print(f"  📊 Prédictions - Length MAE: {results_dict['mae_length_nm_pred']:.2f} nm")
            print(f"  📊 Prédictions - Concentration MAE: {results_dict['mae_concentration_pred']:.2e}")

        if true_successful:
            true_mae_cb_fit = [r[0] for r in true_successful]
            true_diam_errors = [r[1] for r in true_successful]
            true_length_errors = [r[2] for r in true_successful]
            true_conc_errors = [r[3] for r in true_successful]

            # Résultats détaillés vérité terrain
            true_detailed = []
            for i, (mae_cb, diam_err, length_err, conc_err, pred_diam, pred_length, pred_conc, true_diam, true_length, true_conc) in enumerate(true_successful):
                true_detailed.append({
                    'sample_index': i,
                    'type': 'prediction',
                    'true_diameter_nm': true_diam,
                    'true_length_nm': true_length,
                    'true_concentration': true_conc,
                    'pred_diameter_nm': pred_diam,
                    'pred_length_nm': pred_length,
                    'pred_concentration': pred_conc,
                    'mae_cb': mae_cb,
                    'diameter_abs_error': diam_err,
                    'length_abs_error': length_err,
                    'concentration_abs_error': conc_err
                })

            results_dict.update({
                'sasfit_true_samples': len(true_successful),
                'mae_cb_true': float(np.mean(true_mae_cb_fit)),
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

            print(f"  ✅ Vérité terrain - MAE entre courbes fittées: {results_dict['mae_cb_true']:.2f}")
            print(f"  ✅ Vérité terrain - Diamètre MAE: {results_dict['mae_diameter_nm_true']:.2f} nm")
            print(f"  ✅ Vérité terrain - Length MAE: {results_dict['mae_length_nm_true']:.2f} nm")
            print(f"  ✅ Vérité terrain - Concentration MAE: {results_dict['mae_concentration_true']:.2e}")
            print(f"  ✅ Vérité terrain - Concentration err MIN: {float(np.min(true_conc_errors)):.2e}")
            print(f"  ✅ Vérité terrain - Concentration err MAX: {float(np.max(true_conc_errors)):.2e}")

        # Calcul du ratio de performance (prédiction / vérité terrain)
        if pred_successful and true_successful:
            mae_cb_fitted_ratio = results_dict['mae_cb_pred'] / results_dict['mae_cb_true']
            diam_ratio = results_dict['mae_diameter_nm_pred'] / results_dict['mae_diameter_nm_true']
            length_ratio = results_dict['mae_length_nm_pred'] / results_dict['mae_length_nm_true']
            conc_ratio = results_dict['mae_concentration_pred'] / results_dict['mae_concentration_true']

            results_dict.update({
                'sasfit_mae_cb_fitted_ratio': float(mae_cb_fitted_ratio),
                'sasfit_diameter_mae_ratio': float(diam_ratio),
                'sasfit_length_mae_ratio': float(length_ratio),
                'sasfit_concentration_mae_ratio': float(conc_ratio)
            })

            print(f"  📈 Ratio performance - MAE courbes fitted: {mae_cb_fitted_ratio:.2f}x (1.0=parfait)")
            print(f"  📈 Ratio performance - Diamètre: {diam_ratio:.2f}x (1.0=parfait)")
            print(f"  📈 Ratio performance - Length: {length_ratio:.2f}x (1.0=parfait)")
            print(f"  📈 Ratio performance - Concentration: {conc_ratio:.2f}x (1.0=parfait)")

        # Sauvegarde détails combinés
        if pred_successful or true_successful:
            all_detailed = []
            if pred_successful:
                all_detailed.extend(pred_detailed)
            if true_successful:
                all_detailed.extend(true_detailed)

            df = pd.DataFrame(all_detailed)
            csv_path = self.output_dir / "sasfit_detailed_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Détails SASFit sauvés: {csv_path}")

        results_dict.update({
            'sasfit_total_processed': len(pred_samples),
            'sasfit_percentage': self.sasfit_percentage,
            'sasfit_transforms_inverted': True
        })

        return results_dict


    def run_validation(self) -> Dict[str, Any]:
        """Exécute le calcul complet des métriques de validation."""
        print(f"🚀 Démarrage validation - Random State: {self.random_state}")

        # Calcul des métriques
        reconstruction_metrics = self.compute_reconstruction_metrics()
        sasfit_metrics = self.compute_sasfit_metrics()

        # Fusion résultats
        results = {**reconstruction_metrics, **sasfit_metrics}
        results.update({
            'random_state': self.random_state,
            'model_type': self.model_type,
            'checkpoint_path': str(self.checkpoint_path),
            'data_path': str(self.data_path)
        })

        # Sauvegarde
        self._save_results(results)

        print(f"\n✅ Validation terminée - Résultats dans {self.output_dir}")
        return results

    def _save_results(self, results: Dict[str, Any]):
        """Sauvegarde les résultats sous différents formats."""
        # JSON
        json_path = self.output_dir / "validation_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # YAML
        yaml_path = self.output_dir / "validation_metrics.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        # Résumé texte
        summary_path = self.output_dir / "metrics_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=== MÉTRIQUES DE VALIDATION ===\n\n")
            f.write(f"Modèle: {results.get('model_type', 'N/A')}\n")
            f.write(f"Échantillons reconstruction: {results.get('samples_evaluated', 0)}\n")
            f.write(f"Échantillons SASFit: {results.get('sasfit_samples', 0)}\n")
            f.write(f"Random state: {results.get('random_state', 'N/A')}\n\n")

            if 'global_mae' in results:
                f.write("RECONSTRUCTION:\n")
                f.write(f"  MAE: {results['global_mae']:.6f}\n")
                f.write(f"  RMSE: {results.get('global_rmse', 'N/A'):.6f}\n")
                f.write(f"  R²: {results.get('global_r2', 'N/A'):.6f}\n\n")

            if 'mae_diameter_nm' in results:
                f.write("SASFIT:\n")
                f.write(f"  Diamètre MAE: {results['mae_diameter_nm']:.2f} nm\n")
                f.write(f"  Concentration MAE: {results['mae_concentration']:.2e}\n")

        print(f"✓ Résultats sauvés: JSON, YAML, résumé")


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Calcul optimisé des métriques de validation")

    parser.add_argument("-c", "--checkpoint", required=True, help="Chemin checkpoint modèle")
    parser.add_argument("-d", "--data_path", required=True, help="Chemin fichier HDF5")
    parser.add_argument("-o", "--outputdir", required=True, help="Répertoire de sortie")
    parser.add_argument("--signal_length", type=int, help="Longueur de signal forcée", default=300)
    parser.add_argument("-cd", "--conversion_dict", help="Fichier conversion métadonnées")

    parser.add_argument("--batch_size", type=int, default=32, help="Taille batch")
    parser.add_argument("--eval_percentage", type=float, default=0.1,
                       help="% dataset pour métriques reconstruction")
    parser.add_argument("--sasfit_percentage", type=float, default=0.005,
                       help="% dataset pour SASFit")

    parser.add_argument("--qmin_fit", type=float, default=0.001, help="Q min fitting")
    parser.add_argument("--qmax_fit", type=float, default=0.3, help="Q max fitting")
    parser.add_argument("--factor_scale_to_conc", type=float, default=20878,
                       help="Facteur conversion échelle->concentration")

    parser.add_argument("--n_processes", type=int, help="Nombre processus SASFit")
    parser.add_argument("--random_state", type=int, default=42, help="Graine aléatoire")

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()

    print("🔬 Calculateur de Métriques de Validation v2.0")
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
        sasfit_percentage=args.sasfit_percentage,
        qmin_fit=args.qmin_fit,
        qmax_fit=args.qmax_fit,
        factor_scale_to_conc=args.factor_scale_to_conc,
        n_processes=args.n_processes,
        random_state=args.random_state,
        signal_length=args.signal_length
    )

    calculator.run_validation()


if __name__ == "__main__":
    main()