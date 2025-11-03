"""Training pipeline that orchestrates datasets, models, and callbacks."""

from importlib import import_module, util
from pathlib import Path
import os

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split
from uniqpath import unique_path

from src.model.pairvae.configs import PairVAEModelConfig, PairVAETrainingConfig
from src.model.vae.configs import VAETrainingConfig, VAEModelConfig
from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetPairH5 import PairHDF5Dataset
from src.dataset.transformations import Pipeline
from src.dataset.utils import build_subset
from src.model.callbacks.inference_callback import InferencePlotCallback
from src.model.callbacks.metrics_callback import MAEMetricCallback
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE
from src.logging_utils import get_logger

torch.set_float32_matmul_precision('high')

logger = get_logger(__name__)

class NumpySafeDumper(yaml.SafeDumper):
    pass

def np_representer(dumper, data):
    return dumper.represent_list(data.tolist())

NumpySafeDumper.add_representer(np.ndarray, np_representer)

class TrainPipeline:
    """High-level orchestration class that instantiates data, model, and trainer."""

    def __init__(self, config: dict, verbose=False):
        """Validate configuration, prepare datasets, and instantiate the trainer."""
        self.verbose = verbose
        self.cfg_model = config.get('model')
        self.config = self._set_defaults(config)
        if self.verbose:
            logger.info("Loading configuration")
            logger.info(
                "Configuration:\n%s",
                yaml.dump(
                    self.config,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    Dumper=NumpySafeDumper
                ),
            )
            logger.info("Building components")
        self.log_path = self._safe_log_directory()
        self.config['training']['output_dir'] = str(self.log_path)
        self.model, self.dataset, self.extra_callback_list = self._initialize_components()
        self.config['conversion_dict'] = self.dataset.get_conversion_dict()
        if self.verbose:
            logger.info("Preparing data loaders")
        self.training_loader, self.validation_loader, self.test_dataloader = self._create_data_loaders()
        if self.verbose:
            logger.info("Building trainer")
        self.trainer = self._configure_trainer()
        self.model.save_hyperparameters()
        if self.verbose:
            logger.info("Preparing log directory")
        self.log_directory = self._setup_log_directory()
        self.model.set_global_config(self.config)

    def _set_defaults(self, config):
        """Validate the configuration and set sensible defaults."""
        for key in ['experiment_name', 'run_name', 'model', 'dataset', 'training']:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        training = config['training']
        for key in ['num_epochs']:
            if key not in training:
                raise ValueError(f"Missing required training config key: {key}")

        training.setdefault('patience', max(1, training['num_epochs'] // 5))
        training.setdefault('batch_size', 32)
        training.setdefault('use_loglog', True)
        training.setdefault('num_gpus', 1)
        training.setdefault('num_nodes', 1)
        training.setdefault('save_every', 1)
        training.setdefault('output_dir', 'train_results')
        training.setdefault('plot_train', True)
        training.setdefault('every_n_epochs', 10)
        training.setdefault('num_samples', 10)
        training.setdefault('weighted_loss', False)
        config['training'] = training

        if 'type' not in config['model']:
            raise ValueError("Missing required model type in config['model']['type']")
        return config

    def _safe_log_directory(self) -> Path:
        """Create and return a unique log directory path."""
        if 'output_dir' not in self.config['training']:
            log_path = unique_path(Path(self.config['experiment_name'], self.config['run_name']))
            log_path.mkdir(parents=True)
        else:
            base_path = Path(self.config['training']['output_dir'], self.config['experiment_name'])
            base_path.mkdir(parents=True, exist_ok=True)
            log_path = unique_path(base_path / self.config['run_name'])
            log_path.mkdir(parents=True)
        return log_path

    def _initialize_components(self):
        """
        Instantiate the model, dataset, and callbacks according to the current configuration.

        Returns:
            tuple:
                model (pl.LightningModule): Lightning-wrapped model (PlVAE or PlPairVAE)
                dataset (torch.utils.data.Dataset): Dataset object for training/validation
                callbacks (list): List of Lightning callbacks
        """

        model_type = self.config["model"]["type"].lower()
        training_cfg = self.config["training"]
        dataset_cfg = self.config["dataset"]
        output_dir = self.log_path / "inference_results"
        common_cb_args = {
            "num_samples": training_cfg.get("num_samples", 10),
            "every_n_epochs": training_cfg.get("every_n_epochs", 10),
            "artifact_file": "val_plot.png",
        }
        callbacks = []

        # ----------------------- PairVAE ---------------------------------
        if model_type == "pair_vae":
            model_cfg = PairVAEModelConfig(**self.config["model"])
            train_cfg = PairVAETrainingConfig(**training_cfg)
            model = PlPairVAE(model_config=model_cfg, train_config=train_cfg)
            transform_les = model.model.vae_les.get_transformer()
            transform_saxs = model.model.vae_saxs.get_transformer()


            dataset = PairHDF5Dataset(
                **dataset_cfg,
                transformer_q_saxs=Pipeline(transform_saxs["q"]),
                transformer_y_saxs=Pipeline(transform_saxs["y"]),
                transformer_q_les=Pipeline(transform_les["q"]),
                transformer_y_les=Pipeline(transform_les["y"]),
                use_data_q=False,
            )

            curves_config = {
                "saxs": {"truth_key": "data_y_saxs", "pred_keys": ["recon_saxs", "recon_les2saxs"], "use_loglog": True},
                "les": {"truth_key": "data_y_les", "pred_keys": ["recon_les", "recon_saxs2les"]},
            }

        # ----------------------- Single VAE ---------------------------------
        elif model_type == "vae":
            model_cfg = VAEModelConfig(**self.config["model"])
            train_cfg = VAETrainingConfig(**training_cfg)

            transforms = self.config.get("transforms_data", {})
            assert "q" in transforms and "y" in transforms, "Missing 'q' or 'y' in transform config."

            ds_cfg = dict(dataset_cfg)
            required_meta = set(ds_cfg.get("requested_metadata") or [])
            required_meta.update({"diameter_nm", "concentration_original"})
            ds_cfg["requested_metadata"] = sorted(required_meta)
            dataset = HDF5Dataset(
                **ds_cfg,
                transformer_q=Pipeline(transforms["q"]),
                transformer_y=Pipeline(transforms["y"]),
            )
            model_cfg.data_q = dataset.get_data_q()
            model_cfg.transforms_data = dataset.transforms_to_dict()

            model = PlVAE(model_config=model_cfg, 
                          train_config=train_cfg)
            curves_config = {
                "recon": {
                    "truth_key": "data_y",
                    "pred_keys": ["recon"],
                    "use_loglog": training_cfg.get("use_loglog", False),
                }
            }
            from src.model.callbacks.metrics_callback import MAEMetricCallback
            callbacks += [MAEMetricCallback()]
            try:
                from src.model.callbacks.metrics_callback import SASFitMetricCallback
                callbacks += [SASFitMetricCallback()]
            except Exception:
                pass

        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'vae' or 'pair_vae'.")

        # ----------------------- Callbacks setup ---------------------------------
        inf_val = InferencePlotCallback(
            curves_config=curves_config,
            output_dir=output_dir,
            **common_cb_args,
        )
        callbacks.append(inf_val)

        if training_cfg.get("plot_train", False):
            train_args = {**common_cb_args, "artifact_file": "train_plot.png"}
            inf_train = InferencePlotCallback(
                curves_config=curves_config,
                output_dir=output_dir,
                **train_args,
            )
            callbacks.insert(0, inf_train)
        if hasattr(dataset, "transforms_to_dict"):
            self.config["transforms_data"] = dataset.transforms_to_dict()
        return model, dataset, callbacks

    def _create_data_loaders(self):
        """Create train/validation/test DataLoaders depending on provided index arrays."""
        cfg_train = self.config['training']

        train_indices_path = cfg_train.get('array_train_indices')
        val_indices_path = cfg_train.get('array_val_indices')
        test_indices_path = cfg_train.get('array_test_indices')

        loaders = {'train': None, 'val': None, 'test': None}
        subsets = {}

        if train_indices_path and val_indices_path:
            train_indices_raw = np.load(train_indices_path, allow_pickle=True) if train_indices_path else None
            val_indices_raw = np.load(val_indices_path, allow_pickle=True) if val_indices_path else None
            test_indices_raw = np.load(test_indices_path, allow_pickle=True) if test_indices_path else None

            model_type = self.cfg_model['type'].lower()
            model_spec = self.cfg_model.get('spec', 'not defined').lower()

            def extract_indices(data, pos, split_name):
                try:
                    return [pair[pos] for pair in data]
                except Exception as e:
                    raise ValueError(
                        f"Failed to extract {split_name} indices at position {pos}. "
                        f"Expected each element to be an iterable with at least {pos + 1} entries. "
                        f"Received structure: {type(data)} (length={len(data)}). "
                        f"Error details: {e}. "
                        f"Check the content and shape of the file used for '{split_name}' indices."
                    ) from e

            if model_type == 'vae':
                if model_spec == 'saxs':
                    pos = 1
                elif model_spec == 'les':
                    pos = 2
                elif model_type == 'not defined':
                    raise ValueError("Model spec is 'not defined'; cannot determine index extraction position. choice : ['saxs', 'les', 'pair]")
                else:
                    raise ValueError("For VAE, cfg_model['spec'] must be 'saxs' or 'les'.")
            elif model_type == 'pair' or model_type == 'pair_vae':
                pos = 0
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            train_indices = extract_indices(train_indices_raw, pos, 'train')
            val_indices = extract_indices(val_indices_raw, pos, 'validation')
            test_indices = None
            if test_indices_raw is not None:
                test_indices = extract_indices(test_indices_raw, pos, 'test')

            if train_indices is None:
                raise AssertionError("Train indices must be provided if using index arrays.")
            if val_indices is None:
                raise AssertionError("Validation indices must be provided when train indices are given.")

            subsets['train'] = build_subset(self.dataset, train_indices)
            subsets['val'] = build_subset(self.dataset, val_indices)
            if test_indices is not None:
                subsets['test'] = build_subset(self.dataset, test_indices)

        else:
            logger.warning("No index arrays provided; using 80/20 random split for training and validation.")
            total_samples = len(self.dataset)
            train_count = int(0.8 * total_samples)
            val_count = total_samples - train_count
            subsets['train'], subsets['val'] = random_split(self.dataset, [train_count, val_count])
            if test_indices_path:
                logger.warning("Test indices array provided without train/val indices; ignoring test split.")

        batch_size = cfg_train['batch_size']
        num_workers = cfg_train.get('num_workers', max(1, os.cpu_count()))

        logger.info("Building DataLoaders (batch_size=%d, num_workers=%d)", batch_size, num_workers)

        if 'train' in subsets:
            print("TRAIN SIZE")
            print(len(subsets['train']))  
            loaders['train'] = DataLoader(subsets['train'], batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers)
        if 'val' in subsets:
            loaders['val'] = DataLoader(subsets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if 'test' in subsets:
            loaders['test'] = DataLoader(subsets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return loaders['train'], loaders['val'], loaders['test']

    def _configure_trainer(self):
        """Configure callbacks, logging, and return a Lightning trainer."""
        if self.verbose:
            logger.info("Configuring callbacks and logger")
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=self.config['training']['patience'],
                                            min_delta=self.config['training'].get('min_delta', 0.0000001), verbose=self.verbose,
                                            mode='min')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                              every_n_epochs=self.config['training'].get('save_every', 1),
                                              dirpath=self.log_path,
                                              filename="best")
        if "mlflow_uri" in self.config and self.config["mlflow_uri"] is not None:
            from dotenv import load_dotenv
            load_dotenv()
            self.logger = MLFlowLogger(
                experiment_name=self.config['experiment_name'],
                run_name=self.config['run_name'],
                tracking_uri=self.config.get("mlflow_uri", f"file:{self.config['experiment_name']}/mlrun")
            )
        else:
            from lightning.pytorch.loggers import TensorBoardLogger
            self.logger = TensorBoardLogger(
                save_dir=str(self.log_path.parent / "tensorboard_logs"),
                version=self.config['run_name']
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
            logger=self.logger,
            enable_progress_bar=False,
        )

    def _setup_log_directory(self) -> str:
        """Persist configuration artifacts and log index arrays."""
        file_path = self.log_path / "config_model.yaml"
        with file_path.open("w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True, Dumper=NumpySafeDumper)

        if self.verbose:
            logger.info("Fichier YAML sauvegardé dans : %s", file_path)

        if hasattr(self.trainer.logger, 'experiment') and hasattr(self.trainer.logger.experiment, 'log_artifact'):
            self.trainer.logger.experiment.log_artifact(
                local_path=str(file_path),
                run_id=self.trainer.logger.run_id
            )

        np.save(self.log_path / 'train_indices.npy', self.training_loader.dataset.indices)
        np.save(self.log_path / 'val_indices.npy', self.validation_loader.dataset.indices)
        if self.test_dataloader is not None:
            np.save(self.log_path / 'test_indices.npy', self.test_dataloader.dataset.indices)
        if self.verbose:
            logger.info(
                "Indices sauvegardés dans : %s et %s",
                self.log_path / 'train_indices.npy',
                self.log_path / 'val_indices.npy',
            )

        if hasattr(self.trainer.logger, 'experiment') and hasattr(self.trainer.logger.experiment, 'log_artifact'):
            self.trainer.logger.experiment.log_artifact(local_path=str(self.log_path / 'train_indices.npy'),
                                                        run_id=self.trainer.logger.run_id)
            self.trainer.logger.experiment.log_artifact(local_path=str(self.log_path / 'val_indices.npy'),
                                                        run_id=self.trainer.logger.run_id)

    def train(self):
        """Launch the Lightning training loop and return the log path."""
        logger.info("Starting training")
        self.trainer.fit(
            self.model,
            train_dataloaders=self.training_loader,
            val_dataloaders=self.validation_loader
        )
        logger.info("Training completed")
        if self.test_dataloader is not None :
            logger.info("Starting testing")
            self.trainer.test(self.model, self.test_dataloader, ckpt_path="best")
            logger.info("Testing completed")
        return self.log_path

