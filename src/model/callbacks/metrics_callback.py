import lightning.pytorch as pl
import torch
import torch.nn as nn
import numpy as np
from lmfit import minimize, Parameters
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import empty_data1D
import multiprocessing as mp
from functools import partial


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


def fit_single_sample(args):
    """Fonction pour fitter un seul échantillon - utilisée par multiprocessing"""
    sample_data, qmin_fit, qmax_fit, factor = args
    y_np, q_np, t_d, t_c = sample_data

    # Apply q-range mask
    mask = (q_np >= qmin_fit) & (q_np <= qmax_fit)
    if not np.any(mask):
        return None

    q_fit = q_np[mask]
    i_fit = y_np[mask]

    # Load model for this process
    cyl_model = load_model("cylinder")

    # SLDs for Ag in H2O (Å^-2)
    sld_particle = 7.76211e11 * 1e-16
    sld_solvent = 9.39845e10 * 1e-16

    # Setup fitting
    data_fit = empty_data1D(q_fit)
    calc_fit = DirectModel(data_fit, cyl_model)

    # Parameters
    params = Parameters()
    params.add("scale", value=1e14/factor, min=1e6/factor, max=1e20/factor)
    params.add("radius", value=250.0, min=90.0, max=510.0)
    params.add("length", value=100.0, min=40.0, max=160.0)
    params.add("background", value=0, vary=False)

    def residual_log(p):
        pv = p.valuesdict()
        I = calc_fit(scale=pv["scale"], background=pv["background"],
                     radius=pv["radius"], length=pv["length"],
                     sld=sld_particle, sld_solvent=sld_solvent,
                     radius_pd=0.0, length_pd=0.0)
        eps = 1e-30
        return np.log10(np.clip(i_fit, eps, None)) - np.log10(np.clip(I, eps, None))

    try:
        # Optimisation hybride
        res = minimize(
            residual_log, params,
            method="differential_evolution",
            minimizer_kws=dict(
                popsize=500,
                maxiter=150,
                polish=True,
                tol=1e-5,
                updating="deferred",
                workers=1
            )
        )

        # Local polish
        res = minimize(
            residual_log, res.params,
            method="least_squares",
            loss="linear",
            f_scale=0.1,
            xtol=1e-6, ftol=1e-6, gtol=1e-6,
            max_nfev=40000
        )

        # Extract results
        fitted_scale = res.params["scale"].value
        radius_A = res.params["radius"].value

        # Convert to physical units
        pred_conc = float(fitted_scale * factor)
        radius_nm = float(np.rint(radius_A / 10.0))
        pred_diam_nm = float(radius_nm * 2.0)

        # Calculate errors
        diam_err = abs(pred_diam_nm - t_d)
        conc_err = abs(pred_conc - t_c)

        return (diam_err, conc_err, pred_diam_nm, pred_conc, t_d, t_c)

    except Exception as e:
        return None


class MAEMetricCallback(pl.Callback):
    """Callback to compute and log MAE for each artifact with 'recon' in its key on validation set."""

    def __init__(self):
        self.mae_loss = nn.L1Loss()
        self.best_mae = {}  # Stores best MAE for each recon key

    def on_validation_epoch_end(self, trainer, pl_module):
        mae_dict = {}
        pl_module.eval()
        val_dataloader = trainer.val_dataloaders
        with torch.no_grad():
            for batch in val_dataloader:
                batch = move_to_device(batch, pl_module.device)
                data_y = batch["data_y"]
                inputs = data_y.squeeze(1)
                outputs = pl_module(batch)
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if "recon" in key:
                            recon = value
                            if isinstance(recon, torch.Tensor) and recon.ndim > inputs.ndim:
                                recon = recon.squeeze(1)
                            mae_value = self.mae_loss(recon, inputs).item()
                            mae_dict.setdefault(key, []).append(mae_value)
                else:
                    # Fallback if output is not dict, assume key "recon"
                    recon = outputs[1] if isinstance(outputs, tuple) else outputs
                    if isinstance(recon, torch.Tensor) and recon.ndim > inputs.ndim:
                        recon = recon.squeeze(1)
                    mae_value = self.mae_loss(recon, inputs).item()
                    mae_dict.setdefault("recon", []).append(mae_value)
        metrics = {}
        for key, values in mae_dict.items():
            avg_mae = sum(values) / len(values)
            if key not in self.best_mae or avg_mae < self.best_mae[key]:
                self.best_mae[key] = avg_mae
            metrics[f"val_mae_{key}"] = avg_mae
            metrics[f"best_val_mae_{key}"] = self.best_mae[key]
        trainer.logger.log_metrics(metrics, step=trainer.current_epoch)
        pl_module.train()


class SASFitMetricCallback(pl.Callback):
    """Fit cylinder model to predicted I(q) to recover diameter_nm and concentration, then compare to ground truth."""

    def __init__(self, qmin_fit=0.001, qmax_fit=0.3, val_percentage=0.1, factor_scale_to_conc=20878, n_processes=None):
        super().__init__()
        self.qmin_fit = qmin_fit
        self.qmax_fit = qmax_fit
        self.val_percentage = val_percentage  # Pourcentage du dataset de validation à utiliser
        self.factor = factor_scale_to_conc
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)  # Utilise tous les CPU sauf 1
        self.best = {"diameter_mae": float("inf"), "concentration_mae": float("inf")}

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        val_loader = trainer.val_dataloaders

        # Collecter tous les échantillons du dataset de validation
        all_samples = []

        print(f"Collecte des prédictions sur GPU...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = move_to_device(batch, pl_module.device)
                outputs = pl_module(batch)

                # Extract reconstruction
                if isinstance(outputs, dict):
                    recon = outputs.get('recon', None)
                else:
                    recon = outputs[1] if isinstance(outputs, tuple) else outputs

                if recon is None:
                    continue

                # Prepare data
                y_pred = recon.detach().cpu()
                if y_pred.ndim == 3 and y_pred.size(1) == 1:
                    y_pred = y_pred.squeeze(1)

                q_batch = batch["data_q"].detach().cpu()
                if q_batch.ndim == 3 and q_batch.size(1) == 1:
                    q_batch = q_batch.squeeze(1)

                # Check metadata
                meta = batch.get("metadata", {})
                if not (isinstance(meta, dict) and 'diameter_nm' in meta and 'concentration_original' in meta):
                    continue

                diam_true = meta['diameter_nm'].detach().cpu().numpy()
                conc_true = meta['concentration_original'].detach().cpu().numpy()

                # Collect samples for multiprocessing
                B = y_pred.shape[0]
                for i in range(B):
                    y_np = y_pred[i].numpy()
                    q_np = q_batch[i].numpy()
                    t_d = float(diam_true[i]) if np.ndim(diam_true) > 0 else float(diam_true)
                    t_c = float(conc_true[i]) if np.ndim(conc_true) > 0 else float(conc_true)

                    all_samples.append((y_np, q_np, t_d, t_c))

        # Calculer le nombre d'échantillons à traiter selon le pourcentage
        total_samples = len(all_samples)
        n_samples = int(total_samples * self.val_percentage)

        if n_samples == 0:
            print("Aucun échantillon à traiter!")
            trainer.logger.log_metrics({"sasfit_samples": 0}, step=trainer.current_epoch)
            pl_module.train()
            return

        # Prendre un échantillon aléatoire du pourcentage demandé
        np.random.seed(42)  # Pour la reproductibilité
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
        selected_samples = [all_samples[i] for i in sample_indices]

        print(f"Début du fitting SAS avec {self.n_processes} processus sur {n_samples}/{total_samples} échantillons ({self.val_percentage*100:.1f}%)...")

        # Préparer les arguments pour multiprocessing
        fit_args = [(sample, self.qmin_fit, self.qmax_fit, self.factor) for sample in selected_samples]

        # Multiprocessing fitting
        diam_abs_err = []
        conc_abs_err = []
        count = 0

        with mp.Pool(processes=self.n_processes) as pool:
            results = pool.map(fit_single_sample, fit_args)

            for result in results:
                if result is not None:
                    diam_err, conc_err, pred_diam_nm, pred_conc, t_d, t_c = result
                    diam_abs_err.append(diam_err)
                    conc_abs_err.append(conc_err)
                    count += 1

                    # Progress indicator
                    if count % 50 == 0:
                        print(f"Traité {count}/{n_samples} échantillons...")

        # Log results
        if count > 0 and len(diam_abs_err) > 0 and len(conc_abs_err) > 0:
            diam_mae = float(sum(diam_abs_err) / len(diam_abs_err))
            conc_mae = float(sum(conc_abs_err) / len(conc_abs_err))

            if diam_mae < self.best["diameter_mae"]:
                self.best["diameter_mae"] = diam_mae
            if conc_mae < self.best["concentration_mae"]:
                self.best["concentration_mae"] = conc_mae

            print(f"SASFit terminé: {count}/{n_samples} échantillons réussis, MAE diamètre: {diam_mae:.2f} nm, MAE concentration: {conc_mae:.2e}")

            trainer.logger.log_metrics({
                "val_mae_diameter_nm": diam_mae,
                "best_val_mae_diameter_nm": self.best["diameter_mae"],
                "val_mae_concentration": conc_mae,
                "best_val_mae_concentration": self.best["concentration_mae"],
                "sasfit_samples": count,
                "sasfit_total_available": total_samples,
                "sasfit_percentage_used": self.val_percentage
            }, step=trainer.current_epoch)
        else:
            print("Aucun échantillon SASFit traité avec succès!")
            trainer.logger.log_metrics({"sasfit_samples": 0}, step=trainer.current_epoch)

        pl_module.train()
