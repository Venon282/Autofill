"""Training pipelines for VAE and PairVAE models."""
from enum import Enum
from pathlib import Path
import os
import yaml
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import json
from pydantic.json import pydantic_encoder
from torch.utils.data import DataLoader, random_split
from uniqpath import unique_path

from src.model.callbacks.metrics_callback import MAEMetricCallback
from src.model.configs import (
    VAETrainingConfig, VAEModelConfig, HDF5DatasetConfig,
    PairVAEModelConfig, PairVAETrainingConfig, PairHDF5DatasetConfig,
    ModelType, ModelSpec
)
from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetPairH5 import PairHDF5Dataset
from src.dataset.transformations import Pipeline
from src.dataset.utils import build_subset
from src.model.callbacks.inference_callback import InferencePlotCallback
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE
from src.logging_utils import get_logger

logger = get_logger(__name__, custom_name="TRAINER")
torch.set_float32_matmul_precision("high")

os.environ["NCCL_IB_DISABLE"] = "1" # Force NCCL to ignore network interfaces that might cause issues
os.environ["NCCL_DEBUG"] = "INFO" # Print why it crash

class EpochLogger(Callback):
    """Print a single line of metrics at the end of each epoch, Keras style."""
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Get only epoch-level metrics
        metrics = trainer.callback_metrics
        
        formatted_metrics = []
        for k, v in metrics.items():
            # Remove the step values metrics
            if "_step" in k:
                continue
            
            if isinstance(v, torch.Tensor):
                v = v.item()
                
            formatted_metrics.append(f"{k}: {v}")

        metrics_str = " - ".join(formatted_metrics)
        epoch_str = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        
        print(f"{epoch_str} - {metrics_str}", flush=True)
        
#region Base Train Pipeline
class BaseTrainPipeline:
    """Base pipeline handling shared logic for model training."""

    def __init__(self, model_cfg, train_cfg, dataset_cfg, run_name="run" , verbose=False, experiment_name="train_exp", mlflow_uri=None, show_progressbar=True):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.dataset_cfg = dataset_cfg
        self.run_name = run_name
        self.verbose = verbose
        self.experiment_name = experiment_name or "experiment"
        self.mlflow_uri = mlflow_uri
        self.log_path = self._safe_log_directory()
        self.train_cfg.output_dir = str(self.log_path)
        self.model = None
        self.dataset = None
        self.extra_callback_list = []
        self.training_loader = None
        self.validation_loader = None
        self.test_dataloader = None
        self.trainer = None
        self.show_progressbar = show_progressbar

    def _safe_log_directory(self) -> Path:
        base = Path(self.train_cfg.output_dir or "train_results", self.experiment_name)
        base.mkdir(parents=True, exist_ok=True)
        path = unique_path(base / self.run_name)
        path.mkdir(parents=True)
        return path

    def _configure_trainer(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.train_cfg.patience,
            min_delta=self.train_cfg.min_delta,
            verbose=self.verbose,
            mode="min",
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            every_n_epochs=self.train_cfg.save_every,
            dirpath=self.log_path,
            filename="best",
        )

        if self.mlflow_uri:
            from dotenv import load_dotenv
            load_dotenv()
            logger_obj = MLFlowLogger(
                experiment_name=self.experiment_name,
                run_name=self.run_name,
                tracking_uri=self.mlflow_uri,
            )
        else:
            logger_obj = TensorBoardLogger(
                save_dir=str(self.log_path.parent / "tensorboard_logs"),
                name=self.experiment_name,
                version=self.run_name,
            )
        


        logger_obj.log_hyperparams(self.train_cfg.model_dump(exclude={"verbose"}))
        logger_obj.log_hyperparams(self.model_cfg.model_dump(exclude={"transforms_data", "data_q", "verbose"}))
        logger_obj.log_hyperparams(self.dataset_cfg.model_dump(exclude={"transforms_data", "verbose"}))

        callbacks = [early_stop, checkpoint] + self.extra_callback_list
        
        # If no progress bar print one line by epochs
        if not self.show_progressbar:
            callbacks.append(EpochLogger())
            
        if self.train_cfg.num_gpus > 1:
            strategy = DDPStrategy(
                process_group_backend="nccl", 
                find_unused_parameters=True, # Set to True if VAE have conditional branches
                start_method="spawn" 
            )
        else:
            strategy = "auto"
            

        #strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
        accelerator = "auto" #"gpu" if torch.cuda.is_available() else "cpu"
        devices = self.train_cfg.num_gpus if self.train_cfg.num_gpus > 0 else 1 #torch.cuda.is_available() else 1

        return pl.Trainer(
            strategy=strategy,
            accelerator=accelerator,
            devices=devices,
            num_nodes=self.train_cfg.num_nodes,
            max_epochs=self.train_cfg.num_epochs,
            log_every_n_steps=10,
            callbacks=callbacks,
            logger=logger_obj,
            enable_progress_bar=self.show_progressbar,
            precision="32",
        )

    def _create_data_loaders(self):
        cfg = self.train_cfg
        loaders, subsets = {"train": None, "val": None, "test": None}, {}
        if getattr(cfg, "train_indices_path", None) and getattr(cfg, "val_indices_path", None):
            train_idx = np.load(cfg.train_indices_path, allow_pickle=True)
            val_idx = np.load(cfg.val_indices_path, allow_pickle=True)
            test_idx = np.load(cfg.test_indices_path, allow_pickle=True) if getattr(cfg, "test_indices_path", None) else None
            pos = self._resolve_index_position()

            def extract_indices(data): return [pair[pos] for pair in data]

            subsets["train"] = build_subset(self.dataset, extract_indices(train_idx), sample_frac=cfg.sample_frac)
            subsets["val"] = build_subset(self.dataset, extract_indices(val_idx), sample_frac=cfg.sample_frac)
            if test_idx is not None:
                subsets["test"] = build_subset(self.dataset, extract_indices(test_idx), sample_frac=cfg.sample_frac)
        else:
            total = len(self.dataset)
            train_count = int(0.8 * total)
            subsets["train"], subsets["val"] = random_split(self.dataset, [train_count, total - train_count])

        bs = cfg.batch_size
        nw =  min(cfg.num_workers, os.cpu_count())
        loaders["train"] = DataLoader(subsets["train"], batch_size=bs, shuffle=True, num_workers=nw)
        loaders["val"] = DataLoader(subsets["val"], batch_size=bs, shuffle=False, num_workers=nw)
        if "test" in subsets and subsets["test"]:
            loaders["test"] = DataLoader(subsets["test"], batch_size=bs, shuffle=False, num_workers=nw)
        return loaders["train"], loaders["val"], loaders.get("test")

    # region Save
    def _save_indices(self):
        def _safe_save(loader, name):
            path = self.log_path / f"{name}_indices.npy"
            if hasattr(loader.dataset, "indices"):
                np.save(path, loader.dataset.indices)
            else:
                np.save(path, np.arange(len(loader.dataset)))
        _safe_save(self.training_loader, "train")
        _safe_save(self.validation_loader, "val")
        if self.test_dataloader is not None:
            _safe_save(self.test_dataloader, "test")

    def _save_config(self, filename: str = "config.yaml") -> Path:
        """
        Save the complete resolved configuration (model, training, dataset)
        to YAML and JSON files.
        """
        cfg = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "model": serialize_config(self.model_cfg),
            "training": serialize_config(self.train_cfg),
            "dataset": serialize_config(self.dataset_cfg),
            "mlflow_uri": self.mlflow_uri,
        }

        path = self.log_path / filename
        np.save(self.log_path / 'train_indices.npy', self.training_loader.dataset.indices)
        np.save(self.log_path / 'val_indices.npy', self.validation_loader.dataset.indices)
        if self.test_dataloader is not None:
            np.save(self.log_path / 'test_indices.npy', self.test_dataloader.dataset.indices)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        if hasattr(self.trainer.logger, 'experiment') and hasattr(self.trainer.logger.experiment, 'log_artifact'):
            self.trainer.logger.experiment.log_artifact(
                local_path=str(self.log_path / 'train_indices.npy'),
                run_id=self.trainer.logger.run_id
            )
            self.trainer.logger.experiment.log_artifact(
                local_path=str(self.log_path / 'val_indices.npy'),
                run_id=self.trainer.logger.run_id
            )
            self.trainer.logger.experiment.log_artifact(
                local_path=str(path),
                run_id=self.trainer.logger.run_id
            )
            if self.test_dataloader is not None:
                self.trainer.logger.experiment.log_artifact(
                    local_path=str(self.log_path / 'test_indices.npy'),
                    run_id=self.trainer.logger.run_id
                )

        json_path = path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(cfg, jf, indent=2, ensure_ascii=False, default=str)

        logger.info("Configuration saved to %s and %s", path, json_path)
        return path
    # endregion

    def _resolve_index_position(self):
        raise NotImplementedError

    def _initialize_components(self):
        raise NotImplementedError

    def train(self):
        self._initialize_components()
        self.training_loader, self.validation_loader, self.test_dataloader = self._create_data_loaders()
        self.trainer = self._configure_trainer()
        self._save_indices()
        self._save_config()
        self.trainer.fit(self.model, train_dataloaders=self.training_loader, val_dataloaders=self.validation_loader)
        if self.test_dataloader:
            self.trainer.test(self.model, self.test_dataloader, ckpt_path="best")
        return self.log_path

    def _init_callbacks(self, curves: dict):
        callbacks = []
        if self.train_cfg.plot_train or self.train_cfg.plot_val:
            callbacks.append(
                InferencePlotCallback(
                    curves_config=curves,
                    num_samples=self.train_cfg.num_samples,
                    every_n_epochs=self.train_cfg.every_n_epochs,
                    plot_on_train=self.train_cfg.plot_train,
                    plot_on_val=self.train_cfg.plot_val,
                    output_dir=self.log_path / "inference_results",
                )
            )
        if self.model_cfg.type == ModelType.VAE.value:
            callbacks.append(MAEMetricCallback())
        return callbacks
#endregion


#region Single VAE Pipeline
class SingleVAEPipeline(BaseTrainPipeline):
    """Training pipeline for single-domain VAE."""

    def _initialize_components(self):
        transforms = self.dataset_cfg.transforms_data
        assert "q" in transforms and "y" in transforms, "Missing transform keys."
        dataset = HDF5Dataset(
            hdf5_file=self.dataset_cfg.hdf5_file,
            conversion_dict=self.dataset_cfg.conversion_dict,
            metadata_filters=self.dataset_cfg.metadata_filters,
            requested_metadata=self.dataset_cfg.requested_metadata,
            transformer_q=Pipeline(transforms["q"]),
            transformer_y=Pipeline(transforms["y"]),
            use_data_q=self.dataset_cfg.use_data_q,
            show_progressbar = self.show_progressbar
        )
        self.model_cfg.data_q = dataset.get_data_q()
        self.model_cfg.transforms_data = dataset.transforms_to_dict()
        model = PlVAE(model_config=self.model_cfg, train_config=self.train_cfg)
        model.set_global_config(
                {"experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "model": serialize_config(self.model_cfg),
            "training": serialize_config(self.train_cfg),
            "dataset": serialize_config(self.dataset_cfg),
            "mlflow_uri": self.mlflow_uri,})
        curves = {"recon": {"truth_key": "data_y", "pred_keys": ["recon"], "use_loglog": self.train_cfg.use_loglog}}

        self.model, self.dataset, self.extra_callback_list = model, dataset, self._init_callbacks(curves)

    def _resolve_index_position(self):
        if self.model_cfg.spec == ModelSpec.SAXS:
            return 1
        elif self.model_cfg.spec == ModelSpec.LES:
            return 2
        raise ValueError(f"Invalid spec for VAE: {self.model_cfg.spec}")
#endregion


#region Pair VAE Pipeline
class PairVAEPipeline(BaseTrainPipeline):
    """Training pipeline for paired-domain VAE."""

    def _initialize_components(self):
        model = PlPairVAE.from_pretrained_subvaes(
            model_config=self.model_cfg,
            train_config=self.train_cfg,
        )
        transform_les = model.model.vae_les.get_transformer()
        transform_saxs = model.model.vae_saxs.get_transformer()
        dataset = PairHDF5Dataset(
            hdf5_file=self.dataset_cfg.hdf5_file,
            conversion_dict=self.dataset_cfg.conversion_dict,
            metadata_filters=self.dataset_cfg.metadata_filters,
            requested_metadata=self.dataset_cfg.requested_metadata,
            transformer_q_saxs=Pipeline(transform_saxs["q"]),
            transformer_y_saxs=Pipeline(transform_saxs["y"]),
            transformer_q_les=Pipeline(transform_les["q"]),
            transformer_y_les=Pipeline(transform_les["y"]),
            use_data_q=self.dataset_cfg.use_data_q,
            show_progressbar=self.show_progressbar
        )
        model.set_global_config(
                {"experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "model": serialize_config(self.model_cfg),
            "training": serialize_config(self.train_cfg),
            "dataset": serialize_config(self.dataset_cfg),
            "mlflow_uri": self.mlflow_uri,})
        curves = {
            "saxs": {"truth_key": "data_y_saxs", "pred_keys": ["recon_saxs", "recon_les2saxs"], "use_loglog": True},
            "les": {"truth_key": "data_y_les", "pred_keys": ["recon_les", "recon_saxs2les"]},
        }
        self.model, self.dataset, self.extra_callback_list = model, dataset, self._init_callbacks(curves)

    def _resolve_index_position(self):
        return 0
#endregion


#region Pipeline Factory
def make_trainer(config: dict, verbose=False, show_progressbar=False) -> BaseTrainPipeline:
    """Factory to create the appropriate training pipeline."""
    required_keys = ["model", "training", "dataset", "experiment_name"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config section(s): {missing}. "
                       f"Expected keys: {required_keys}")

    try:
        model_type = ModelType(config["model"]["type"].lower())
    except KeyError:
        raise KeyError("Missing key 'type' in config['model']. "
                       "Expected 'vae' or 'pair_vae'.")
    except ValueError as e:
        raise ValueError(f"Invalid model type '{config['model'].get('type')}'. "
                         f"Valid types: {[m.value for m in ModelType]}. Error: {e}")

    try:
        if model_type == ModelType.VAE:
            return SingleVAEPipeline(
                model_cfg=VAEModelConfig(**config["model"]),
                train_cfg=VAETrainingConfig(**config["training"]),
                dataset_cfg=HDF5DatasetConfig(**config["dataset"]),
                run_name=config["run_name"] if "run_name" in config else "run",
                verbose=verbose,
                experiment_name=config["experiment_name"],
                mlflow_uri=config.get("mlflow_uri"),
                show_progressbar=show_progressbar,
            )

        elif model_type == ModelType.PAIR_VAE:
            return PairVAEPipeline(
                model_cfg=PairVAEModelConfig(**config["model"]),
                train_cfg=PairVAETrainingConfig(**config["training"]),
                dataset_cfg=PairHDF5DatasetConfig(**config["dataset"]),
                run_name=config["run_name"] if "run_name" in config else "run",
                verbose=verbose,
                experiment_name=config["experiment_name"],
                mlflow_uri=config.get("mlflow_uri"),
                show_progressbar=show_progressbar,
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    except TypeError as e:
        raise TypeError(f"Unexpected field or wrong type in configuration. {e}")

    except ValueError as e:
        raise ValueError(f"Configuration validation failed. {e}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error while creating pipeline: {e}")
#endregion

def serialize_config(obj):
    """Recursively convert Pydantic models, enums, and numpy arrays to serializable formats."""
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'model_dump'):
        # Pydantic v2
        data = obj.model_dump(mode='python')
        return serialize_config(data)
    elif hasattr(obj, 'dict'):
        # Pydantic v1
        data = obj.dict()
        return serialize_config(data)
    elif isinstance(obj, dict):
        return {k: serialize_config(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_config(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj