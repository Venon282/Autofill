import argparse
import sys
import os
import torch
import yaml
import numpy as np
import json
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.inferencer import BaseInferencer, move_to_device
from src.model.callbacks.metrics_callback import SASFitMetricCallback, MAEMetricCallback
from src.dataset.datasetH5 import HDF5Dataset
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE


class MetricsInferencer(BaseInferencer):
    """Inférenceur pour calculer les métriques de validation sur un dataset H5"""

    def __init__(self, output_dir, checkpoint_path, hparams, data_path, conversion_dict_path=None,
                 batch_size=32, qmin_fit=0.001, qmax_fit=0.3,
                 eval_percentage=1.0, sasfit_percentage=0.1, factor_scale_to_conc=20878,
                 n_processes=None, random_state=42):

        # Paramètres pour l'évaluation
        self.eval_percentage = eval_percentage
        self.sasfit_percentage = sasfit_percentage
        self.random_state = random_state

        # Initialiser les callbacks pour les métriques
        self.mae_callback = MAEMetricCallback()
        self.sasfit_callback = SASFitMetricCallback(
            qmin_fit=qmin_fit,
            qmax_fit=qmax_fit,
            val_percentage=1.0,  # On gère nous-mêmes l'échantillonnage
            factor_scale_to_conc=factor_scale_to_conc,
            n_processes=n_processes
        )

        # Pas de sauvegarde de plots pour les métriques
        super().__init__(output_dir, False, checkpoint_path, hparams, data_path,
                        conversion_dict_path, sample_frac=1.0, batch_size=batch_size)

        self.results = {}
        self.detailed_results = []

    def load_model(self, path):
        """Charge le modèle depuis le checkpoint"""
        model_type = self.config['model']['type'].lower()
        if model_type == 'vae':
            return PlVAE.load_from_checkpoint(checkpoint_path=path)
        elif model_type == 'pair_vae':
            return PlPairVAE.load_from_checkpoint(checkpoint_path=path)
        else:
            raise ValueError(f"Model type {model_type} is not supported.")

    def get_input_dim(self):
        """Retourne la dimension d'entrée du modèle"""
        model_type = self.config['model']['type'].lower()
        if model_type == 'vae':
            return self.config["model"]["args"]["input_dim"]
        else:
            return None

    def compute_dataset(self, input_dim):
        """Configure le dataset H5 pour les métriques"""
        if not self.data_path.endswith(".h5"):
            raise ValueError("MetricsInferencer only supports HDF5 files (.h5)")

        self.dataset = HDF5Dataset(
            self.data_path,
            sample_frac=1.0,  # Toujours charger le dataset complet
            transformer_q=self.config["transforms_data"]["q"],
            transformer_y=self.config["transforms_data"]["y"],
            metadata_filters=self.config["dataset"]["metadata_filters"],
            conversion_dict=self.conversion_dict,
            requested_metadata=['shape', 'material', 'concentration', 'dimension1',
                              'dimension2', 'opticalPathLength', 'd', 'h',
                              'diameter_nm', 'concentration_original']
        )
        self.format = 'h5'
        self.invert = self.dataset.invert_transforms_func()

    def compute_detailed_metrics(self):
        """Calcule les métriques détaillées avec l'échantillonnage spécifié"""
        print(f"Calcul des métriques détaillées sur {self.eval_percentage*100:.1f}% du dataset...")

        # Set random seed
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # Calculer le nombre d'échantillons à traiter
        total_samples = len(self.dataset)
        samples_to_process = int(total_samples * self.eval_percentage)

        # Générer les indices aléatoirement
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
        selected_indices = all_indices[:samples_to_process]

        print(f"Évaluation de {samples_to_process}/{total_samples} échantillons...")

        # Créer un subset du dataset
        from torch.utils.data import Subset
        subset_dataset = Subset(self.dataset, selected_indices)
        loader = DataLoader(subset_dataset, batch_size=self.batch_size, shuffle=False)

        all_predictions = []
        all_true_values = []
        processed_count = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing detailed metrics"):
                batch = move_to_device(batch, self.device)
                data_y = batch["data_y"]
                true_values = data_y.squeeze(1)

                outputs = self.model(batch)

                # Extraire la reconstruction
                if isinstance(outputs, dict):
                    recon = outputs.get('recon', None)
                    if recon is None:
                        for key, value in outputs.items():
                            if "recon" in key:
                                recon = value
                                break
                else:
                    recon = outputs[1] if isinstance(outputs, tuple) else outputs

                if recon is None:
                    continue

                # Ajuster les dimensions
                if isinstance(recon, torch.Tensor) and recon.ndim > true_values.ndim:
                    recon = recon.squeeze(1)

                # Convertir en numpy
                recon_np = recon.detach().cpu().numpy()
                true_np = true_values.detach().cpu().numpy()

                # Gérer les indices
                batch_indices = selected_indices[processed_count:processed_count + recon_np.shape[0]]

                # Métadonnées
                metadata_batch = batch.get("metadata", {})

                # Stocker les résultats pour chaque échantillon
                for i in range(recon_np.shape[0]):
                    sample_idx = int(batch_indices[i])
                    pred_values = recon_np[i].flatten()
                    true_vals = true_np[i].flatten()

                    # Calculer les métriques par échantillon
                    mae_sample = mean_absolute_error(true_vals, pred_values)
                    mse_sample = mean_squared_error(true_vals, pred_values)
                    r2_sample = r2_score(true_vals, pred_values)

                    # Extraire les métadonnées pour cet échantillon
                    sample_metadata = {}
                    for key, values in metadata_batch.items():
                        if hasattr(values, '__getitem__') and len(values) > i:
                            val = values[i]
                            if hasattr(val, 'item'):
                                val = val.item()
                            sample_metadata[key] = val

                    # Ajouter aux résultats détaillés
                    detailed_row = {
                        'sample_index': sample_idx,
                        'mae': mae_sample,
                        'mse': mse_sample,
                        'r2_score': r2_sample,
                        'rmse': np.sqrt(mse_sample)
                    }

                    # Ajouter les métadonnées
                    detailed_row.update(sample_metadata)
                    self.detailed_results.append(detailed_row)

                    # Stocker pour les métriques globales
                    all_predictions.extend(pred_values)
                    all_true_values.extend(true_vals)

                processed_count += recon_np.shape[0]

        # Calculer les métriques globales
        if all_predictions and all_true_values:
            all_pred_np = np.array(all_predictions)
            all_true_np = np.array(all_true_values)

            global_mae = mean_absolute_error(all_true_np, all_pred_np)
            global_mse = mean_squared_error(all_true_np, all_pred_np)
            global_rmse = np.sqrt(global_mse)
            global_r2 = r2_score(all_true_np, all_pred_np)

            # Métriques additionnelles
            global_mean_pred = np.mean(all_pred_np)
            global_mean_true = np.mean(all_true_np)
            global_std_pred = np.std(all_pred_np)
            global_std_true = np.std(all_true_np)

            metrics_results = {
                'global_mae': float(global_mae),
                'global_mse': float(global_mse),
                'global_rmse': float(global_rmse),
                'global_r2_score': float(global_r2),
                'mean_prediction': float(global_mean_pred),
                'mean_true': float(global_mean_true),
                'std_prediction': float(global_std_pred),
                'std_true': float(global_std_true),
                'samples_evaluated': len(self.detailed_results),
                'total_data_points': len(all_predictions),
                'eval_percentage_used': self.eval_percentage,
                'random_state_used': self.random_state
            }

            print(f"\nMétriques globales calculées sur {len(self.detailed_results)} échantillons:")
            print(f"  MAE global: {global_mae:.6f}")
            print(f"  MSE global: {global_mse:.6f}")
            print(f"  RMSE global: {global_rmse:.6f}")
            print(f"  R² global: {global_r2:.6f}")

            return metrics_results

        return {"samples_evaluated": 0}

    def compute_sasfit_metrics(self):
        """Calcule les métriques SASFit uniquement sur le pourcentage spécifié"""
        print(f"Calcul des métriques SASFit sur {self.sasfit_percentage*100:.1f}% du dataset...")

        # Set random seed
        np.random.seed(self.random_state)

        # Calculer le nombre d'échantillons pour SASFit
        total_samples = len(self.dataset)
        sasfit_samples = int(total_samples * self.sasfit_percentage)

        if sasfit_samples == 0:
            print("Aucun échantillon à traiter pour SASFit!")
            return {"sasfit_samples": 0}

        # Échantillonnage aléatoire direct
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
        sasfit_indices = all_indices[:sasfit_samples]

        print(f"Traitement de {sasfit_samples}/{total_samples} échantillons pour SASFit...")

        # Créer un subset pour SASFit
        from torch.utils.data import Subset
        sasfit_dataset = Subset(self.dataset, sasfit_indices)
        loader = DataLoader(sasfit_dataset, batch_size=self.batch_size, shuffle=False)

        all_samples = []
        sasfit_detailed = []
        processed_count = 0

        # Collecter uniquement les échantillons nécessaires
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting SASFit samples"):
                batch = move_to_device(batch, self.device)
                outputs = self.model(batch)

                # Extraire la reconstruction
                if isinstance(outputs, dict):
                    recon = outputs.get('recon', None)
                    if recon is None:
                        for key, value in outputs.items():
                            if "recon" in key:
                                recon = value
                                break
                else:
                    recon = outputs[1] if isinstance(outputs, tuple) else outputs

                if recon is None:
                    continue

                # Préparer les données
                y_pred = recon.detach().cpu()
                if y_pred.ndim == 3 and y_pred.size(1) == 1:
                    y_pred = y_pred.squeeze(1)

                q_batch = batch["data_q"].detach().cpu()
                if q_batch.ndim == 3 and q_batch.size(1) == 1:
                    q_batch = q_batch.squeeze(1)

                # Vérifier les métadonnées
                meta = batch.get("metadata", {})
                if not (isinstance(meta, dict) and 'diameter_nm' in meta and 'concentration_original' in meta):
                    continue

                diam_true = meta['diameter_nm'].detach().cpu().numpy()
                conc_true = meta['concentration_original'].detach().cpu().numpy()

                # Récupérer les indices originaux
                batch_indices = sasfit_indices[processed_count:processed_count + y_pred.shape[0]]

                # Collecter les échantillons
                for i in range(y_pred.shape[0]):
                    y_np = y_pred[i].numpy()
                    q_np = q_batch[i].numpy()
                    t_d = float(diam_true[i]) if np.ndim(diam_true) > 0 else float(diam_true)
                    t_c = float(conc_true[i]) if np.ndim(conc_true) > 0 else float(conc_true)
                    sample_idx = int(batch_indices[i])

                    all_samples.append((y_np, q_np, t_d, t_c, sample_idx))

                processed_count += y_pred.shape[0]

        if not all_samples:
            print("Aucun échantillon SASFit valide collecté!")
            return {"sasfit_samples": 0}

        print(f"Fitting SAS sur {len(all_samples)} échantillons...")

        # Utiliser la fonction de fitting du callback
        from src.model.callbacks.metrics_callback import fit_single_sample
        import multiprocessing as mp

        fit_args = [(sample[:4], self.sasfit_callback.qmin_fit,
                    self.sasfit_callback.qmax_fit, self.sasfit_callback.factor)
                   for sample in all_samples]

        diam_abs_err = []
        conc_abs_err = []
        count = 0

        with mp.Pool(processes=self.sasfit_callback.n_processes) as pool:
            results = pool.map(fit_single_sample, fit_args)

            for idx, result in enumerate(results):
                if result is not None:
                    diam_err, conc_err, pred_diam_nm, pred_conc, t_d, t_c = result
                    sample_idx = all_samples[idx][4]  # L'index était le 5ème élément

                    diam_abs_err.append(diam_err)
                    conc_abs_err.append(conc_err)

                    # Ajouter aux résultats détaillés SASFit
                    sasfit_detailed.append({
                        'sample_index': sample_idx,
                        'true_diameter_nm': t_d,
                        'pred_diameter_nm': pred_diam_nm,
                        'true_concentration': t_c,
                        'pred_concentration': pred_conc,
                        'diameter_abs_error': diam_err,
                        'concentration_abs_error': conc_err
                    })

                    count += 1

        # Sauvegarder les détails SASFit en CSV
        if sasfit_detailed:
            df_sasfit = pd.DataFrame(sasfit_detailed)
            sasfit_csv = os.path.join(self.output_dir, "sasfit_detailed_results.csv")
            df_sasfit.to_csv(sasfit_csv, index=False)
            print(f"Résultats SASFit détaillés sauvegardés dans {sasfit_csv}")

        # Calculer les résultats
        sasfit_results = {
            "sasfit_samples": count,
            "sasfit_total_processed": len(all_samples),
            "sasfit_percentage_used": self.sasfit_percentage
        }

        if count > 0 and diam_abs_err and conc_abs_err:
            diam_mae = float(sum(diam_abs_err) / len(diam_abs_err))
            conc_mae = float(sum(conc_abs_err) / len(conc_abs_err))

            # Calculer MSE et RMSE pour SASFit
            diam_mse = float(sum(e**2 for e in diam_abs_err) / len(diam_abs_err))
            conc_mse = float(sum(e**2 for e in conc_abs_err) / len(conc_abs_err))

            sasfit_results.update({
                "mae_diameter_nm": diam_mae,
                "mae_concentration": conc_mae,
                "mse_diameter_nm": diam_mse,
                "mse_concentration": conc_mse,
                "rmse_diameter_nm": float(np.sqrt(diam_mse)),
                "rmse_concentration": float(np.sqrt(conc_mse))
            })

            print(f"SASFit terminé: {count}/{len(all_samples)} échantillons réussis")
            print(f"Diamètre - MAE: {diam_mae:.2f} nm, MSE: {diam_mse:.2f}, RMSE: {np.sqrt(diam_mse):.2f}")
            print(f"Concentration - MAE: {conc_mae:.2e}, MSE: {conc_mse:.2e}, RMSE: {np.sqrt(conc_mse):.2e}")

        return sasfit_results

    def infer_and_save(self):
        """Calcule et sauvegarde les métriques"""
        print(f"Évaluation avec eval_percentage={self.eval_percentage*100:.1f}%, sasfit_percentage={self.sasfit_percentage*100:.1f}% (random_state={self.random_state})")

        # Calculer les métriques détaillées
        detailed_results = self.compute_detailed_metrics()
        self.results.update(detailed_results)

        # Calculer les métriques SASFit
        sasfit_results = self.compute_sasfit_metrics()
        self.results.update(sasfit_results)

        # Sauvegarder les résultats globaux
        results_file = os.path.join(self.output_dir, "validation_metrics.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Sauvegarder en YAML pour lisibilité
        results_yaml = os.path.join(self.output_dir, "validation_metrics.yaml")
        with open(results_yaml, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)

        # Sauvegarder les résultats détaillés en CSV
        if self.detailed_results:
            df = pd.DataFrame(self.detailed_results)
            csv_file = os.path.join(self.output_dir, "reconstruction_metrics_detailed.csv")
            df.to_csv(csv_file, index=False)
            print(f"Métriques de reconstruction détaillées sauvegardées dans {csv_file}")

        # Créer un fichier de résumé des métriques
        summary_file = os.path.join(self.output_dir, "metrics_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== RÉSUMÉ DES MÉTRIQUES DE VALIDATION ===\n\n")
            f.write(f"Dataset - Eval: {self.eval_percentage*100:.1f}% ({self.results.get('samples_evaluated', 0)} échantillons)\n")
            f.write(f"Dataset - SASFit: {self.sasfit_percentage*100:.1f}% ({self.results.get('sasfit_samples', 0)} échantillons)\n")
            f.write(f"Random state: {self.random_state}\n\n")

            f.write("MÉTRIQUES DE RECONSTRUCTION:\n")
            f.write(f"  MAE global: {self.results.get('global_mae', 'N/A'):.6f}\n")
            f.write(f"  MSE global: {self.results.get('global_mse', 'N/A'):.6f}\n")
            f.write(f"  RMSE global: {self.results.get('global_rmse', 'N/A'):.6f}\n")
            f.write(f"  R² global: {self.results.get('global_r2_score', 'N/A'):.6f}\n\n")

            if 'sasfit_samples' in self.results and self.results['sasfit_samples'] > 0:
                f.write("MÉTRIQUES SASFIT:\n")
                f.write(f"  Échantillons traités: {self.results['sasfit_samples']}\n")
                f.write(f"  Diamètre MAE: {self.results.get('mae_diameter_nm', 'N/A'):.2f} nm\n")
                f.write(f"  Diamètre RMSE: {self.results.get('rmse_diameter_nm', 'N/A'):.2f} nm\n")
                f.write(f"  Concentration MAE: {self.results.get('mae_concentration', 'N/A'):.2e}\n")
                f.write(f"  Concentration RMSE: {self.results.get('rmse_concentration', 'N/A'):.2e}\n")

        print(f"\nRésultats des métriques de validation:")
        for key, value in self.results.items():
            if isinstance(value, float):
                if 'mae' in key.lower() or 'mse' in key.lower() or 'rmse' in key.lower():
                    print(f"  {key}: {value:.6f}")
                elif 'r2' in key.lower():
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

        print(f"\nFichiers sauvegardés dans {self.output_dir}:")
        print(f"  - reconstruction_metrics_detailed.csv (métriques par échantillon)")
        print(f"  - sasfit_detailed_results.csv (résultats SASFit détaillés)")
        print(f"  - validation_metrics.json (métriques globales)")
        print(f"  - metrics_summary.txt (résumé lisible)")


def parse_args():
    parser = argparse.ArgumentParser(description="Calcul des métriques de validation sur un dataset H5")
    parser.add_argument("-o", "--outputdir", type=str, required=True,
                        help="Répertoire de sortie pour les résultats")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint du modèle")
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Chemin vers le fichier HDF5")
    parser.add_argument("-cd", "--conversion_dict", type=str, required=False,
                        help="Chemin vers le fichier de conversion des métadonnées")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="Taille du batch")
    parser.add_argument("--qmin_fit", type=float, default=0.001,
                        help="Q minimum pour le fitting SAS")
    parser.add_argument("--qmax_fit", type=float, default=0.3,
                        help="Q maximum pour le fitting SAS")
    parser.add_argument("--eval_percentage", type=float, default=1.0,
                        help="Pourcentage du dataset à utiliser pour l'évaluation des métriques générales")
    parser.add_argument("--sasfit_percentage", type=float, default=0.1,
                        help="Pourcentage du dataset à utiliser pour SASFit")
    parser.add_argument("--factor_scale_to_conc", type=float, default=20878,
                        help="Facteur de conversion échelle vers concentration")
    parser.add_argument("--n_processes", type=int, default=None,
                        help="Nombre de processus pour SASFit (défaut: CPU-1)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="État aléatoire pour la reproductibilité")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Charger les hyperparamètres depuis le checkpoint
    ckpt = args.checkpoint
    hparams = torch.load(ckpt, map_location='cpu')['hyper_parameters']['config']
    model_type = hparams['model']['type']

    print(f"Calcul des métriques de validation pour le modèle {model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.data_path}")
    print(f"Pourcentage évaluation: {args.eval_percentage*100:.1f}%")
    print(f"Pourcentage SASFit: {args.sasfit_percentage*100:.1f}%")
    print(f"Random state: {args.random_state}")

    # Créer l'inférenceur de métriques
    runner = MetricsInferencer(
        output_dir=args.outputdir,
        checkpoint_path=args.checkpoint,
        hparams=hparams,
        data_path=args.data_path,
        conversion_dict_path=args.conversion_dict,
        batch_size=args.batch_size,
        qmin_fit=args.qmin_fit,
        qmax_fit=args.qmax_fit,
        eval_percentage=args.eval_percentage,
        sasfit_percentage=args.sasfit_percentage,
        factor_scale_to_conc=args.factor_scale_to_conc,
        n_processes=args.n_processes,
        random_state=args.random_state
    )

    # Exécuter le calcul des métriques
    runner.infer()
    print(f"\nMétriques de validation sauvegardées dans {args.outputdir}")
