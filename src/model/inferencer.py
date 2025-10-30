"""Utilities for running inference from trained VAE and PairVAE checkpoints."""

from __future__ import annotations

import abc
import json
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetTXT import TXTDataset
from src.dataset.transformations import Pipeline
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE
from src.logging_utils import get_logger


logger = get_logger(__name__)


def move_to_device(batch, device: torch.device):
    """Recursively move tensors and collections onto ``device``."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


class BaseInferencer(abc.ABC):
    """Shared scaffolding for dataset preparation and checkpoint loading."""

    def __init__(
        self,
        output_dir: str,
        save_plot: bool,
        checkpoint_path: str,
        hparams: Dict,
        data_path: str,
        conversion_dict_path: Optional[str] = None,
        sample_frac: float = 1.0,
        batch_size: int = 32,
        data_dir: str = ".",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.config = hparams
        self.model = self.load_model(checkpoint_path)
        self.model.to(self.device).eval()
        if conversion_dict_path is None:
            self.conversion_dict = self.config.get("conversion_dict")
        else:
            with open(conversion_dict_path, 'r', encoding='utf-8') as file:
                self.conversion_dict = json.load(file)
        self.data_path = data_path
        self.data_dir = data_dir
        self.sample_frac = sample_frac
        self.save_plot = save_plot
        self.use_loglog = self.model.model_config.use_loglog or False
        self.dataset = None
        self.invert = lambda y, q: (y, q)
        self.format = ''
        self.compute_dataset(self.get_input_dim())
        logger.info("Dataset prepared with %d samples", len(self.dataset))
        logger.info("Device set to %s", self.device)

    @abc.abstractmethod
    def get_input_dim(self) -> Optional[int]:
        """Return the expected model input dimension when applicable."""

    @abc.abstractmethod
    def compute_dataset(self, input_dim: Optional[int]) -> None:
        """Prepare the dataset according to ``data_path`` and ``input_dim``."""

    @abc.abstractmethod
    def load_model(self, path: str):
        """Load and return the Lightning module from ``path``."""

    @abc.abstractmethod
    def infer_and_save(self) -> None:
        """Run inference and persist artifacts to ``output_dir``."""

    def save_pred(self, batch, index: int, q_arr: np.ndarray, y_arrs: Dict[str, np.ndarray], metadata: Dict) -> None:
        """Save predictions and metadata for a single sample."""

        converted_metadata = {}
        for key, value in metadata.items():
            value = value.detach().cpu().numpy().item()
            if self.conversion_dict and key in self.conversion_dict:
                inverse = {v_: k_ for k_, v_ in self.conversion_dict[key].items()}
                value = inverse.get(value, value)
            converted_metadata[key] = value

        name = f"sample_{index}"

        prediction_dir = os.path.join(self.output_dir, f"prediction_{name}")
        os.makedirs(prediction_dir, exist_ok=True)
        for key, y_arr in y_arrs.items():
            txt_filename = os.path.join(prediction_dir, f"prediction_{key}.txt")
            np.savetxt(txt_filename, np.column_stack((q_arr, y_arr)))
            if self.save_plot:
                plot_filename = os.path.join(prediction_dir, f"plot_{key}.png")
                plt.figure()
                if self.use_loglog:
                    plt.loglog(q_arr, y_arr, label=f'{key} prediction')
                else:
                    plt.plot(q_arr, y_arr, label=f'{key} prediction')
                plt.xlabel('q')
                plt.ylabel('y')
                plt.title(f'Prediction {key}')
                plt.grid(True)
                plt.legend()
                plt.savefig(plot_filename)
                plt.close()

        yaml_filename = os.path.join(prediction_dir, "metadata.yaml")
        with open(yaml_filename, "w", encoding="utf-8") as yaml_file:
            yaml.dump(converted_metadata, yaml_file)

    def infer(self) -> None:
        """Execute :meth:`infer_and_save` and report the output location."""
        logger.info("Starting inference...")
        self.infer_and_save()
        logger.info("Inference results saved in %s", self.output_dir)


class VAEInferencer(BaseInferencer):
    """Run inference for single-spectrum VAE checkpoints."""

    def load_model(self, path: str):
        return PlVAE.load_from_checkpoint(checkpoint_path=path)

    def get_input_dim(self) -> int:
        return self.model.model_config.args.input_dim

    def infer_and_save(self) -> None:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference per sample"):
                batch = move_to_device(batch, self.device)
                outputs = self.model(batch)
                y_pred = outputs["recon"]
                q_pred = outputs['data_q']
                y_pred, q_pred = self.invert(y_pred, q_pred)
                metadata_batch = batch["metadata"]
                for i in range(len(y_pred)):
                    y_arr = y_pred[i].detach().cpu().numpy().flatten()
                    q_arr = q_pred[i].detach().cpu().numpy().flatten()
                    metadata = {k: metadata_batch[k][i] for k in metadata_batch}
                    technique = self.config["dataset"]["metadata_filters"].get("technique", ["signal"])
                    if isinstance(technique, str):
                        technique = [technique]
                    signal_name = "_".join(technique)
                    y_arrs = {signal_name or "recon": y_arr}
                    self.save_pred(batch, i, q_arr, y_arrs, metadata)

    def compute_dataset(self, input_dim: Optional[int]) -> None:
        transforms_q = Pipeline(self.model.get_transformer_q())
        transforms_y = Pipeline(self.model.get_transformer_y())
        if self.data_path.endswith(".h5"):
            self.dataset = HDF5Dataset(
                self.data_path,
                sample_frac=self.sample_frac,
                transformer_q=transforms_q,
                transformer_y=transforms_y,
                # metadata_filters=self.config["dataset"]["metadata_filters"],
                conversion_dict=self.conversion_dict,
                use_data_q=False
            )
            self.format = 'h5'
            self.invert = self.dataset.invert_transforms_func()
        elif self.data_path.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(self.data_path)
            # technique_filter = [t.lower() for t in self.config["dataset"]["metadata_filters"].get("technique", [])]
            # material_filter = [m.lower() for m in self.config["dataset"]["metadata_filters"].get("material", [])]
            # if technique_filter:
            #     df = df[df["technique"].str.lower().isin(technique_filter)]
            # if material_filter:
            #     df = df[df["material"].str.lower().isin(material_filter)]
            df = df.reset_index(drop=True)
            self.dataset = TXTDataset(
                dataframe=df,
                data_dir=self.data_dir,
                transformer_q=transforms_q,
                transformer_y=transforms_y,
            )
            self.format = 'csv'
            self.invert = self.dataset.invert_transforms_func()
        else:
            raise ValueError("Unsupported file format. Use .h5 or .csv")


class PairVAEInferencer(BaseInferencer):
    """Run inference for PairVAE checkpoints."""

    def __init__(
        self,
        mode: str,
        output_dir: str,
        save_plot: bool,
        checkpoint_path: str,
        hparams: Dict,
        data_path: str,
        conversion_dict_path: Optional[str] = None,
        sample_frac: float = 1.0,
        batch_size: int = 32,
        data_dir: str = ".",
    ) -> None:
        valid_modes = {'les_to_saxs', 'saxs_to_les', 'les_to_les', 'saxs_to_saxs'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {sorted(valid_modes)}.")
        self.mode = mode
        super().__init__(output_dir, save_plot, checkpoint_path, hparams, data_path, conversion_dict_path, sample_frac, batch_size, data_dir)
        self.use_loglog =  True if self.mode in ["saxs_to_saxs", "les_to_saxs"] else False

    def load_model(self, path: str):
        return PlPairVAE.load_from_checkpoint(checkpoint_path=path)

    def get_input_dim(self) -> Optional[int]:
        return None

    def infer_and_save(self) -> None:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        model_methods = {
            'les_to_saxs': self.model.les_to_saxs,
            'saxs_to_les': self.model.saxs_to_les,
            'les_to_les': self.model.les_to_les,
            'saxs_to_saxs': self.model.saxs_to_saxs,
        }
        with torch.no_grad():
            for batch in tqdm(loader, desc="PairVAE inference"):
                batch = move_to_device(batch, self.device)
                y_pred, q_pred = model_methods[self.mode](batch)
                y_pred, q_pred = self.invert(y_pred, q_pred)
                metadata_batch = batch["metadata"]
                for i in range(len(y_pred)):
                    metadata = {k: metadata_batch[k][i] for k in metadata_batch}
                    y_arr = y_pred[i].detach().cpu().numpy().flatten()
                    q_arr = q_pred
                    y_arrs = {self.mode: y_arr}
                    self.save_pred(batch, i, q_arr, y_arrs, metadata)

    def compute_dataset(self, input_dim: Optional[int]) -> None:
        transform_config = self.config.get('transforms_data', {})
        transformers = {
            'les': {
                'q': Pipeline(transform_config["q_les"]),
                'y': Pipeline(transform_config["y_les"]),
            },
            'saxs': {
                'q': Pipeline(transform_config["q_saxs"]),
                'y': Pipeline(transform_config["y_saxs"]),
            },
        }
        mode_config = {
            'les_to_saxs': {'input': 'les', 'output': 'saxs', 'technique': ['les']},
            'saxs_to_les': {'input': 'saxs', 'output': 'les', 'technique': ['saxs']},
            'les_to_les': {'input': 'les', 'output': 'les', 'technique': ['les']},
            'saxs_to_saxs': {'input': 'saxs', 'output': 'saxs', 'technique': ['saxs']},
        }
        config = mode_config[self.mode]
        input_transformers = transformers[config['input']]
        output_transformers = transformers[config['output']]

        if self.data_path.endswith(".h5") or self.data_path.endswith(".hdf5"):
            self.dataset = HDF5Dataset(
                self.data_path,
                sample_frac=self.sample_frac,
                metadata_filters=self.config["dataset"]["metadata_filters"],
                conversion_dict=self.conversion_dict,
                transformer_q=input_transformers['q'],
                transformer_y=input_transformers['y'],
                requested_metadata=['diameter_nm', 'length_nm', 'concentration_original', 'concentration'],
                    use_data_q=False 
            )
            self.format = 'h5'
        elif self.data_path.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(self.data_path)
            self.dataset = TXTDataset(
                dataframe=df,
                data_dir=self.data_dir,
                transformer_q=input_transformers['q'],
                transformer_y=input_transformers['y'],
            )
            self.format = 'csv'
        else:
            raise ValueError("Unsupported file format. Use .h5 or .csv")

        def invert(y, q):
            y_inv = output_transformers['y'].invert(y.cpu())
            q_inv = output_transformers['q'].invert(q.cpu())
            return y_inv, q_inv

        self.invert = invert

