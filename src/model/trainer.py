from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import random_split, DataLoader
try:
    from uniqpath import unique_path
except Exception:
    def unique_path(path: Path) -> Path:
        p = Path(path)
        if not p.exists():
            return p
        i = 1
        while True:
            candidate = p.parent / f"{p.name}_{i}"
            if not candidate.exists():
                return candidate
            i += 1

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.utils import *
from src.dataset.datasetPairH5 import PairHDF5Dataset
from src.dataset.transformations import Pipeline
from src.model.callbacks.inference_callback import InferencePlotCallback
from src.model.callbacks.metrics_callback import MAEMetricCallback
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE


class TrainPipeline:
    def __init__(self, config: dict, verbose=True):
        self.verbose = verbose
        self.config = self._set_defaults(config)
        if self.verbose:
            print("[Pipeline] Loading configuration")
            print(yaml.dump(self.config, default_flow_style=False, sort_keys=False, allow_unicode=True))
            print("[Pipeline] Building components")
        self.log_path = self._safe_log_directory()
        self.config['training']['output_dir'] = str(self.log_path)
        self.model, self.dataset, self.extra_callback_list = self._initialize_components()
        self.config['conversion_dict'] = self.dataset.get_conversion_dict()
        if self.verbose:
            print("[Pipeline] Preparing data loaders")
        self.training_loader, self.validation_loader = self._create_data_loaders()
        if self.verbose:
            print("[Pipeline] Building trainer")
        self.trainer = self._configure_trainer()
        self.model.save_hyperparameters()
        if self.verbose:
            print("[Pipeline] Preparing log directory")
        self.log_directory = self._setup_log_directory()

    def _set_defaults(self, config):
        # Required fields
        for key in ['experiment_name', 'run_name', 'model', 'dataset', 'training']:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        training = config['training']
        # Required training fields
        for key in ['num_epochs']:
            if key not in training:
                raise ValueError(f"Missing required training config key: {key}")

        # Set intelligent defaults
        training.setdefault('patience', max(1, training['num_epochs'] // 5))
        training.setdefault('batch_size', 32)
        training.setdefault('num_workers', 4)
        training.setdefault('use_loglog', True)
        training.setdefault('num_gpus', 1)
        training.setdefault('save_every', 1)
        training.setdefault('output_dir', 'train_results')
        training.setdefault('plot_train', True)
        training.setdefault('every_n_epochs', 10)
        training.setdefault('num_samples', 10)
        config['training'] = training

        # Model type default
        if 'type' not in config['model']:
            raise ValueError("Missing required model type in config['model']['type']")
        return config

    def _safe_log_directory(self) -> Path:
        if 'output_dir' not in self.config['training']:
            log_path = unique_path(Path(self.config['experiment_name'], self.config['run_name']))
            log_path.mkdir(parents=True)
        else:
            base_path = Path(self.config['training']['output_dir'])
            base_path.mkdir(parents=True, exist_ok=True)
            log_path = unique_path(base_path / self.config['run_name'])
            log_path.mkdir(parents=True)
        return log_path

    def _initialize_components(self):
        model_type = self.config['model']['type']
        common_cfg = {
            'num_samples': self.config['training'].get('num_samples', 10),
            'every_n_epochs': self.config['training'].get('every_n_epochs', 10),
            'artifact_file': 'val_plot.png'
        }
        callbacks = []
        if model_type.lower() == 'pair_vae':
            model = PlPairVAE(self.config)
            transform_les = model.get_transforms_data_les()
            transform_saxs = model.get_transforms_data_saxs()
            print(f"config : {self.config}")
            print("#" * 50)
            dataset = PairHDF5Dataset(**self.config['dataset'],
                                      transformer_q_saxs=Pipeline(transform_saxs["q"]),
                                      transformer_y_saxs=Pipeline(transform_saxs["y"]),
                                      transformer_q_les=Pipeline(transform_les["q"]),
                                      transformer_y_les=Pipeline(transform_les["y"]))
            curves_config = {
                'saxs': {'truth_key': 'data_y_saxs', 'pred_keys': ['recon_saxs', 'recon_les2saxs'], 'use_loglog': True},
                'les': {'truth_key': 'data_y_les', 'pred_keys': ['recon_les', 'recon_saxs2les']}
            }
        elif model_type.lower() == 'vae':
            model = PlVAE(self.config)
            transform_config = self.config.get('transforms_data', {})
            assert 'q' in transform_config and 'y' in transform_config, "Missing 'q' or 'y' in transform config"
            # Ensure required metadata for SAS fitting is requested
            ds_cfg = dict(self.config['dataset'])
            req_meta = list(ds_cfg.get('requested_metadata', []) or [])
            for k in ['diameter_nm', 'concentration_original']:
                if k not in req_meta:
                    req_meta.append(k)
            ds_cfg['requested_metadata'] = req_meta
            dataset = HDF5Dataset(**ds_cfg,
                                  transformer_q=Pipeline(transform_config["q"]),
                                  transformer_y=Pipeline(transform_config["y"]))
            curves_config = {'recon': {'truth_key': 'data_y', 'pred_keys': ["recon"],
                                       'use_loglog': self.config['training']['use_loglog']}}
            callbacks.append(MAEMetricCallback())
            # Lazy import to keep static analyzer happy
            try:
                from src.model.callbacks.metrics_callback import SASFitMetricCallback
                callbacks.append(SASFitMetricCallback())
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'pair' or 'vae'.")

        inf_cb = InferencePlotCallback(
            curves_config=curves_config,
            output_dir=self.config['training'].get('output_dir', 'inference_results'),
            **common_cfg
        )
        callbacks.append(inf_cb)
        if self.config['training'].get('plot_train', False):
            train_cfg = common_cfg.copy()
            train_cfg['artifact_file'] = 'train_plot.png'
            callbacks.insert(0, InferencePlotCallback(curves_config=curves_config,
                                                      output_dir=str(self.log_path / "inference_results"),
                                                      **train_cfg))

        # Save transformer config
        self.config['transforms_data'] = dataset.transforms_to_dict()

        return model, dataset, callbacks

    def _create_data_loaders_old(self):
        """
        TODO
        fonction de séparation selon les critères d'apparaillage, 
        par exemple si critèrev de pair saxs/les est selon concentration/materiel/forme
        on forme tous les trios possible, 80% vont en train, 20% en val
        IL FAUT ENREGISTRERle split pendant les vae simples pour l'utiliser dans le pair
        """
        dataset_instance = self.dataset
        total_samples = len(dataset_instance)
        train_count = int(0.8 * total_samples)
        validation_count = total_samples - train_count
        if self.verbose:
            print(f"[Data] Splitting dataset: train={train_count}, validation={validation_count}")
        train_subset, validation_subset = random_split(dataset_instance, [train_count, validation_count])
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        if self.verbose:
            print(f"[Data] Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
        training_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return training_loader, validation_loader

    def _create_data_loaders(self):
        """
        TODO
        fonction de séparation selon les critères d'apparaillage, 
        par exemple si critèrev de pair saxs/les est selon concentration/materiel/forme
        on forme tous les trios possible, 80% vont en train, 20% en val
        IL FAUT ENREGISTRERle split pendant les vae simples pour l'utiliser dans le pair
        """

        train_csv_indices_saxs_les = np.load(self.config['training']['array_train_indices'], allow_pickle=True)
        val_csv_indices_saxs_les = np.load(self.config['training']['array_val_indices'], allow_pickle=True)

        if self.config['model']['type'].lower() == 'vae':
            # select colonne 1 for saxs, colonne 2 for les
            if  "saxs" in self.config['run_name']:
                train_csv_indices = [pair[1] for pair in train_csv_indices_saxs_les]
                val_csv_indices = [pair[1] for pair in val_csv_indices_saxs_les]
            elif "les" in self.config['run_name'] :
                train_csv_indices = [pair[2] for pair in train_csv_indices_saxs_les]
                val_csv_indices = [pair[2] for pair in val_csv_indices_saxs_les]
        
        elif self.config['model']['type'].lower() == 'pair_vae':
            train_csv_indices = [pair[0] for pair in train_csv_indices_saxs_les]
            val_csv_indices = [pair[0] for pair in val_csv_indices_saxs_les]

        train_dataset = build_subset(self.dataset, train_csv_indices)
        validation_subset   = build_subset(self.dataset, val_csv_indices)

        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        if self.verbose:
            print(f"[Data] Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
        training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return training_loader, validation_loader

    def _configure_trainer(self):
        if self.verbose:
            print("[Trainer] Configuring callbacks and logger")
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=self.config['training']['patience'],
                                            min_delta=self.config['training'].get('min_delta', 0.00001), verbose=True,
                                            mode='min')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                              every_n_epochs=self.config['training'].get('save_every', 1),
                                              dirpath=self.log_path,
                                              filename="best")
        if "mlflow_uri" in self.config:
            from dotenv import load_dotenv
            load_dotenv()

        self.logger = MLFlowLogger(
            experiment_name=self.config['experiment_name'],
            run_name=self.config['run_name'],
            tracking_uri=self.config.get("mlflow_uri", f"file:{self.config['experiment_name']}/mlrun")
        )
        self.logger.log_hyperparams(self.config)
        self.all_callbacks = [early_stop_callback, checkpoint_callback] + self.extra_callback_list
        strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        devices = self.config['training']['num_gpus'] if torch.cuda.is_available() else 1
        return pl.Trainer(
            strategy=strategy,
            accelerator=accelerator,
            devices=devices,
            num_nodes=self.config['training']['num_nodes'],
            max_epochs=self.config['training']['num_epochs'],
            log_every_n_steps=10,
            callbacks=self.all_callbacks,
            logger=self.logger
        )

    def _setup_log_directory(self) -> str:

        file_path = self.log_path / "config_model.yaml"
        with file_path.open("w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)

        if self.verbose:
            print(f"Fichier YAML sauvegardé dans : {file_path}")

        self.trainer.logger.experiment.log_artifact(
            local_path=str(file_path),
            run_id=self.trainer.logger.run_id
        )

        np.save(self.log_path / 'train_indices.npy', self.training_loader.dataset.indices)
        np.save(self.log_path / 'val_indices.npy', self.validation_loader.dataset.indices)
        if self.verbose:
            print(
                f"Indices sauvegardés dans : {self.log_path / 'train_indices.npy'} et {self.log_path / 'val_indices.npy'}")
        self.trainer.logger.experiment.log_artifact(local_path=str(self.log_path / 'train_indices.npy'),
                                                    run_id=self.trainer.logger.run_id)
        self.trainer.logger.experiment.log_artifact(local_path=str(self.log_path / 'val_indices.npy'),
                                                    run_id=self.trainer.logger.run_id)

    def train(self):
        print("[Pipeline] Starting training")
        self.trainer.fit(
            self.model,
            train_dataloaders=self.training_loader,
            val_dataloaders=self.validation_loader
        )
        print("[Pipeline] Training completed")
        return self.log_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configuration = yaml.safe_load(f)
    pipeline = TrainPipeline(configuration)
    pipeline.run()
