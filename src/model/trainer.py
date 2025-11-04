from pathlib import Path
import os
import yaml
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from uniqpath import unique_path
from src.dataset.utils import build_subset
from src.dataset.transformations import Pipeline
from src.model.callbacks.inference_callback import InferencePlotCallback
from src.logging_utils import get_logger
from model.configs import (
    ModelType, ModelSpec,
    VAEModelConfig, VAETrainingConfig, HDF5DatasetConfig,
    PairVAEModelConfig, PairVAETrainingConfig, PairHDF5DatasetConfig,
)
from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetPairH5 import PairHDF5Dataset
from src.model.vae.pl_vae import PlVAE
from src.model.pairvae.pl_pairvae import PlPairVAE

logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")


class BaseTrainPipeline:
    """Base training pipeline with shared setup and trainer logic."""

    def __init__(self, model_cfg, train_cfg, dataset_cfg, *, verbose=False, experiment_name=None, mlflow_uri=None):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.dataset_cfg = dataset_cfg
        self.verbose = verbose
        self.experiment_name = experiment_name or "experiment"
        self.mlflow_uri = mlflow_uri
        self.log_path = self._safe_log_directory()
        self.run_name = self.log_path.name
        self.train_cfg.output_dir = str(self.log_path)
        self.model = None
        self.dataset = None
        self.extra_callback_list = []
        self.training_loader = None
        self.validation_loader = None
        self.test_dataloader = None
        self.trainer = None

    # ---------- shared methods ----------

    def _safe_log_directory(self) -> Path:
        base = Path(self.train_cfg.output_dir or "train_results", self.experiment_name)
        base.mkdir(parents=True, exist_ok=True)
        path = unique_path(base / "run")
        path.mkdir(parents=True)
        return path

    def _configure_trainer(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.train_cfg.patience,
            min_delta=getattr(self.train_cfg, "min_delta", 1e-7),
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

        callbacks = [early_stop, checkpoint] + self.extra_callback_list
        strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = self.train_cfg.num_gpus if torch.cuda.is_available() else 1

        return pl.Trainer(
            strategy=strategy,
            accelerator=accelerator,
            devices=devices,
            num_nodes=self.train_cfg.num_nodes,
            max_epochs=self.train_cfg.num_epochs,
            log_every_n_steps=10,
            callbacks=callbacks,
            logger=logger_obj,
            enable_progress_bar=True,
        )

    def _create_data_loaders(self):
        cfg = self.train_cfg
        loaders, subsets = {"train": None, "val": None, "test": None}, {}
        if cfg.train_indices_path and cfg.val_indices_path:
            train_idx = np.load(cfg.train_indices_path, allow_pickle=True)
            val_idx = np.load(cfg.val_indices_path, allow_pickle=True)
            test_idx = np.load(cfg.test_indices_path, allow_pickle=True) if cfg.test_indices_path else None
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
        nw = getattr(cfg, "num_workers", max(1, os.cpu_count()))
        loaders["train"] = DataLoader(subsets["train"], batch_size=bs, shuffle=True, num_workers=nw)
        loaders["val"] = DataLoader(subsets["val"], batch_size=bs, shuffle=False, num_workers=nw)
        if "test" in subsets and subsets["test"]:
            loaders["test"] = DataLoader(subsets["test"], batch_size=bs, shuffle=False, num_workers=nw)
        return loaders["train"], loaders["val"], loaders.get("test")

    # ---------- abstract sections ----------

    def _resolve_index_position(self):
        raise NotImplementedError

    def _initialize_components(self):
        raise NotImplementedError

    def train(self):
        train_loader, val_loader, test_loader = self._create_data_loaders()
        self.trainer = self._configure_trainer()
        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        if test_loader:
            self.trainer.test(self.model, test_loader, ckpt_path="best")
        return self.log_path


# ===============================================================
# Single VAE Pipeline
# ===============================================================

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
        )
        self.model_cfg.data_q = dataset.get_data_q()
        self.model_cfg.transforms_data = dataset.transforms_to_dict()
        model = PlVAE(model_config=self.model_cfg, train_config=self.train_cfg)
        curves = {"recon": {"truth_key": "data_y", "pred_keys": ["recon"], "use_loglog": self.train_cfg.use_loglog}}
        callbacks = [InferencePlotCallback(curves_config=curves, output_dir=self.log_path / "inference_results")]
        self.model, self.dataset, self.extra_callback_list = model, dataset, callbacks

    def _resolve_index_position(self):
        if self.model_cfg.spec == ModelSpec.SAXS:
            return 1
        elif self.model_cfg.spec == ModelSpec.LES:
            return 2
        raise ValueError(f"Invalid spec for VAE: {self.model_cfg.spec}")


# ===============================================================
# Pair VAE Pipeline
# ===============================================================

class PairVAEPipeline(BaseTrainPipeline):
    """Training pipeline for paired-domain VAE."""

    def _initialize_components(self):
        model = PlPairVAE(model_config=self.model_cfg, train_config=self.train_cfg)
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
        )
        curves = {
            "saxs": {"truth_key": "data_y_saxs", "pred_keys": ["recon_saxs", "recon_les2saxs"], "use_loglog": True},
            "les": {"truth_key": "data_y_les", "pred_keys": ["recon_les", "recon_saxs2les"]},
        }
        callbacks = [InferencePlotCallback(curves_config=curves, output_dir=self.log_path / "inference_results")]
        self.model, self.dataset, self.extra_callback_list = model, dataset, callbacks

    def _resolve_index_position(self):
        return 0
