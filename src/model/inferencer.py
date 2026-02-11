"""Inference module for VAE and PairVAE models following SOLID principles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from src.logging_utils import get_logger
from src.model.configs import ModelType, ModelSpec
from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetTXT import TXTDataset
from src.dataset.transformations import Pipeline

logger = get_logger(__name__)


# region Data Loading
class DatasetLoader:
    """Responsible for loading datasets with appropriate transforms."""

    @staticmethod
    def load_dataset(
        data_path: str,
        conversion_dict_path: Optional[str],
        transformer_q: Pipeline,
        transformer_y: Pipeline,
        data_dir: str = ".",
        show_progressbar: bool = True,
    ):
        """Load dataset (H5 or CSV/TXT) with given Pipeline transformers."""
        if data_path.endswith(('.h5', '.hdf5')):
            return DatasetLoader._load_h5_dataset(data_path, conversion_dict_path, transformer_q, transformer_y, show_progressbar)
        elif data_path.endswith('.csv'):
            return DatasetLoader._load_csv_dataset(data_path, data_dir, transformer_q, transformer_y)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

    @staticmethod
    def _load_h5_dataset(data_path: str, conversion_dict_path: Optional[str], transformer_q: Pipeline, transformer_y: Pipeline, show_progressbar: bool = True):
        """Load HDF5 dataset."""
        return HDF5Dataset(
            hdf5_file=data_path,
            conversion_dict=conversion_dict_path,
            transformer_q=transformer_q,
            transformer_y=transformer_y,
            use_data_q=True,
            sanity_check=False,
            show_progressbar=show_progressbar,
        )

    @staticmethod
    def _load_csv_dataset(data_path: str, data_dir: str, transformer_q: Pipeline, transformer_y: Pipeline):
        """Load CSV/TXT dataset."""
        import pandas as pd

        df = pd.read_csv(data_path)

        return TXTDataset(
            dataframe=df,
            data_dir=Path(data_dir),
            transformer_q=transformer_q,
            transformer_y=transformer_y,
        )

    @staticmethod
    def prepare_dataloader(dataset, batch_size: int) -> DataLoader:
        """Prepare PyTorch DataLoader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
# endregion


# region Model Loading
class ModelLoader:
    """Responsible for loading models from checkpoints."""

    @staticmethod
    def load_model(checkpoint_path: str, hparams: Dict, device: str = 'cuda'):
        """Load model based on checkpoint type."""
        model_type = ModelLoader._extract_model_type(hparams)

        if model_type == ModelType.VAE:
            return ModelLoader._load_vae(checkpoint_path, device)
        elif model_type == ModelType.PAIR_VAE:
            return ModelLoader._load_pairvae(checkpoint_path, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _extract_model_type(hparams: Dict) -> str:
        """Extract model type from checkpoint."""
        if "model_config" in hparams:
            return hparams["model_config"]["type"]
        elif "pairvae_model_config" in hparams:
            return hparams["pairvae_model_config"]["type"]
        else:
            raise ValueError("Cannot determine model type from checkpoint.")

    @staticmethod
    def _load_vae(checkpoint_path: str, device: str):
        """Load VAE model."""
        from src.model.vae.pl_vae import PlVAE
        with torch.serialization.safe_globals([ModelType, ModelSpec]):
            model = PlVAE.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        return model

    @staticmethod
    def _load_pairvae(checkpoint_path: str, device: str):
        """Load PairVAE model."""
        from src.model.pairvae.pl_pairvae import PlPairVAE
        with torch.serialization.safe_globals([ModelType, ModelSpec, np._core.multiarray._reconstruct, np.ndarray, np.dtype]):
            model = PlPairVAE.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        return model
# endregion


# region Inference Engine
class InferenceEngine:
    """Orchestrates inference process and handles inversion of transforms for outputs.

    The engine expects transformers for the output domains (y and q). The transformers are
    instances of `Pipeline` and are used to invert model outputs back to the original space.
    """

    def __init__(self, model, transformer_y: Pipeline, transformer_q: Optional[Pipeline], device: str = 'cuda'):
        """Initialize engine with model and output-domain transformers.

        transformer_y: pipeline used to invert model predictions (y domain).
        transformer_q: pipeline used to invert q outputs when present (may be None).
        """
        self.model = model
        self.transformer_y = transformer_y
        self.transformer_q = transformer_q
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def run_vae_inference(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run inference for a single-domain VAE and return inverted outputs.

        Returns a dict with keys: 'predictions' (y, inverted), 'q' (q inverted when available),
        'latents', 'indices', 'mode'.
        """
        predictions = []
        originals = []
        latents = []
        indices = []
        q_values = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                output = self.model(batch)

                recon = output.get('recon')
                z = output.get('z')

                if recon is None:
                    raise RuntimeError("Model forward did not return 'recon' key")

                predictions.append(recon.cpu().numpy())
                if z is not None:
                    latents.append(z.cpu().numpy())

                q_out = output.get('data_q', None)
                if q_out is None and 'data_q' in batch:
                    q_out = batch['data_q']
                if q_out is not None:
                    q_values.append(q_out.cpu().numpy())

                if 'data_index' in batch:
                    indices.append(batch['data_index'].cpu().numpy())
                    
                originals.append(batch['data_y_untransformed'].cpu().numpy())

        predictions_concat = np.concatenate(predictions, axis=0) if predictions else np.empty((0,))
        originals_concat = np.concatenate(originals, axis=0) if originals else np.empty((0,))
        latents_concat = np.concatenate(latents, axis=0) if latents else None
        indices_concat = np.concatenate(indices, axis=0) if indices else None

        predictions_inverted = self._invert_predictions(predictions_concat)

        q_final = None
        if q_values:
            q_concat = np.concatenate(q_values, axis=0)
            q_final = self._invert_q(q_concat)

        mode_spec = getattr(getattr(self.model, 'model_cfg', None), 'spec', getattr(self.model, 'spec', ModelSpec.SAXS))
        mode_key = f"recon_{mode_spec.value if isinstance(mode_spec, ModelSpec) else str(mode_spec)}"

        return {
            'predictions': predictions_inverted,
            'originals':originals_concat,
            'latents': latents_concat,
            'indices': indices_concat,
            'q': q_final,
            'mode': mode_key,
        }

    def run_pairvae_inference(self, dataloader: DataLoader, mode: str) -> Dict[str, Any]:
        """Run inference for PairVAE for a specific translation mode.

        The PairVAE methods return (prediction_tensor, data_q). We invert predictions using
        the output-domain y-transform and optionally invert q.
        """
        predictions = []
        originals = []
        indices = []
        q_values = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)

                if mode == 'les_to_saxs':
                    pred, q_out = self.model.les_to_saxs(batch)
                elif mode == 'saxs_to_les':
                    pred, q_out = self.model.saxs_to_les(batch)
                elif mode == 'les_to_les':
                    pred, q_out = self.model.les_to_les(batch)
                elif mode == 'saxs_to_saxs':
                    pred, q_out = self.model.saxs_to_saxs(batch)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                predictions.append(pred.cpu().numpy())
                originals.append(batch['data_y_untransformed'].cpu().numpy())

                if q_out is not None:
                    if isinstance(q_out, torch.Tensor):
                        q_values.append(q_out.cpu().numpy())
                    else:
                        q_values.append(np.array(q_out))

                if 'data_index' in batch:
                    indices.append(batch['data_index'].cpu().numpy())

        predictions_concat = np.concatenate(predictions, axis=0) if predictions else np.empty((0,))
        originals_concat = np.concatenate(originals, axis=0) if originals else np.empty((0,))
        indices_concat = np.concatenate(indices, axis=0) if indices else None

        predictions_inverted = self._invert_predictions(predictions_concat)

        q_final = None
        if q_values:
            q_concat = np.concatenate(q_values, axis=0)
            q_final = self._invert_q(q_concat)

        return {
            'predictions': predictions_inverted,
            'originals': originals_concat,
            'indices': indices_concat,
            'q': q_final,
            'mode': mode,
        }

    def _invert_predictions(self, data: np.ndarray) -> np.ndarray:
        inverted = []
        for i in range(len(data)):
            sample = np.squeeze(data[i])
            inverted_sample = self.transformer_y.invert(sample)
            inverted.append(inverted_sample)
        return np.array(inverted)

    def _invert_q(self, data: np.ndarray) -> np.ndarray:
        if self.transformer_q is None:
            return data
        inverted = []
        for i in range(len(data)):
            sample = np.squeeze(data[i])
            inverted_sample = self.transformer_q.invert(sample)
            inverted.append(inverted_sample)
        return np.array(inverted)

    def _move_to_device(self, batch: Dict) -> Dict:
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._move_to_device(value)
            else:
                result[key] = value
        return result
# endregion


# region Output Writers
class OutputWriter:
    """Base class for output writers."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def write(self, results: Dict[str, Any]):
        raise NotImplementedError


class HDF5Writer(OutputWriter):
    """Write results to HDF5 format."""

    def write(self, results: Dict[str, Any]):
        logger.info(f"Saving predictions to HDF5: {self.output_path}")

        with h5py.File(self.output_path, 'w') as f:
            for key, value in results.items():
                if value is not None and key != 'mode':
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value, compression='gzip')

        logger.info(f"Successfully saved to {self.output_path}")


class TXTWriter(OutputWriter):
    """Write results to TXT format (one file per sample) including q and y columns."""

    def __init__(self, output_path: str, n_jobs: int = 8):
        super().__init__(output_path)
        self.n_jobs = n_jobs

    def write(self, results: Dict[str, Any]):
        logger.info(f"Saving predictions to TXT files in: {self.output_path}")

        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = results.get('predictions')
        q_values = results.get('q')

        if predictions is None:
            raise RuntimeError("No predictions found to write")

        n_samples = len(predictions)

        if q_values is None:
            q_values = [None] * n_samples
        elif isinstance(q_values, np.ndarray) and q_values.ndim == 1:
            q_values = np.tile(q_values, (n_samples, 1))
        elif isinstance(q_values, np.ndarray) and q_values.ndim == 2 and q_values.shape[0] == 1 and n_samples > 1:
            q_values = np.tile(q_values, (n_samples, 1))

        for i in range(n_samples):
            pred = np.squeeze(predictions[i])
            q = None if q_values is None else np.squeeze(q_values[i])

            filename = output_dir / f"prediction_{i:06d}.txt"

            if q is None:
                np.savetxt(filename, pred.reshape(-1, 1), fmt='%.6e')
            else:
                pred_len = len(pred)
                q_truncated = q[:pred_len]
                pair = np.vstack([q_truncated, pred]).T
                np.savetxt(filename, pair, fmt='%.6e', header='q y', comments='')

        logger.info(f"Successfully saved {n_samples} TXT files")


class PlotWriter(OutputWriter):
    """Generate and save diagnostic plots. The plotting scale (loglog) is configured at init."""

    def __init__(self, output_path: str, plot_limit: int = 10, use_loglog: bool = False, plot_original: bool = False,
                 data_pair_path:str=None, input_domain:str=None, output_domain:str=None,
                is_pair:bool=False):
        super().__init__(output_path)
        self.plot_limit = plot_limit
        self.use_loglog = use_loglog
        self.plot_original = plot_original
        self.data_pair_path = data_pair_path
        self.is_pair = is_pair
        self.input_domain = input_domain
        self.output_domain=output_domain
        
    def _pairPlot(self,i, indice, prediction, original, mode):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        plot_original = self.plot_original
        use_loglog = self.use_loglog
        if self.input_domain != self.output_domain:
            if self.data_pair_path:
                with h5py.File(self.data_pair_path, 'r') as f:
                    key_input_indexs = 'data_index_'+self.input_domain
                    good_indice = np.where(f[key_input_indexs][:] == indice)[0]
                    key_output_signal = 'data_y_'+ self.output_domain
                    true_original = f[key_output_signal][good_indice][0]
            else:
                self.plot_original = False
                true_original = None
            
        else:
            true_original = original
            
        self._noPairPlot(i, prediction, true_original, mode + ' - ' + self.output_domain,  fig=fig, ax=axs[0])
        if self.input_domain != self.output_domain:
            self.plot_original = False
            self.use_loglog = not self.use_loglog
            self._noPairPlot(i, prediction=original, original=None, mode=mode + ' - ' + self.input_domain,  fig=fig, ax=axs[1])
            
        self.plot_original = plot_original
        self.use_loglog = use_loglog
        return fig
        
        
    def _noPairPlot(self,i, prediction, original, mode,  fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        pred = np.squeeze(prediction)

        if self.use_loglog:
            if self.plot_original:
                ax.loglog(original.ravel(), linewidth=2, label='true')
                ax.loglog(pred, linewidth=2, label='prediction')
            else:
                ax.loglog(pred, linewidth=2)
            ax.set_xscale("log")
        else:
            if self.plot_original:
                ax.plot(original.ravel(), linewidth=2, label='true')
                ax.plot(pred, linewidth=2, label='prediction')
            else:
                ax.plot(pred, linewidth=2)

        ax.set_xlabel('Index')
        ax.set_ylabel('Intensity')
        ax.set_title(f'{mode.capitalize()} - Sample {i}')
        ax.grid(True, alpha=0.3)
        
        if self.plot_original:
            ax.legend()

        return fig

    def write(self, results: Dict[str, Any]):
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = results.get('predictions', [])
        originals = results['originals']
        mode = results.get('mode', 'reconstruction')
        indices = results['indices']

        n_plots = min(self.plot_limit, len(predictions))

        logger.info(f"Generating {n_plots} plots (loglog={self.use_loglog})...")

        for i in range(n_plots):
            if self.is_pair:
                fig = self._pairPlot(
                    i=i,
                    indice=indices[i],
                    prediction=predictions[i],
                    original=originals[i],
                    mode=mode,
                )
            else:
                fig = self._noPairPlot(
                    i=i,
                    prediction=predictions[i],
                    original=originals[i],
                    mode=mode,
                )
            plot_path = output_dir / f"i{i:06d}_{mode}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"Plots saved to {output_dir}")
# endregion


# region Sampling Utility
class DatasetSampler:
    """Handle dataset sampling logic."""

    @staticmethod
    def sample_dataset(
        dataset,
        sample_frac: float = 1.0,
        sample_seed: int = 42,
        indices_path: Optional[str] = None,
        model_spec: Optional[ModelSpec] = None,
    ):
        """Sample dataset either from .npy file or using sample_frac.

        If indices_path is provided, it takes precedence over sample_frac.
        The .npy file can contain either:
        - Simple indices: [idx1, idx2, ...]
        - Triplets: [[pair_idx, saxs_idx, les_idx], ...]

        When triplets are detected and model_spec is provided:
        - ModelSpec.SAXS extracts position 1 (saxs_idx)
        - ModelSpec.LES extracts position 2 (les_idx)
        """
        if indices_path is not None:
            logger.info(f"Loading indices from: {indices_path}")
            loaded_indices = np.load(indices_path, allow_pickle=True)

            if loaded_indices.ndim == 2 and loaded_indices.shape[1] >= 3:
                if model_spec == ModelSpec.SAXS:
                    pos = 1
                    logger.info(f"Extracting SAXS indices (position {pos})")
                elif model_spec == ModelSpec.LES:
                    pos = 2
                    logger.info(f"Extracting LES indices (position {pos})")
                else:
                    pos = -1
                    logger.warning(f"model_spec not provided, using last position (pos={pos})")

                indices = [pair[pos] for pair in loaded_indices]
            elif loaded_indices.ndim > 1:
                indices = loaded_indices.flatten().tolist()
            else:
                indices = loaded_indices.tolist()

            logger.info(f"Using {len(indices)} indices from file")
            return Subset(dataset, indices)

        if sample_frac == 1.0:
            return dataset

        random.seed(sample_seed)
        np.random.seed(sample_seed)

        total_size = len(dataset)

        if sample_frac > 1.0:
            n_samples = min(int(sample_frac), total_size)
        else:
            n_samples = max(1, int(total_size * sample_frac))

        indices = random.sample(range(total_size), n_samples)
        logger.info(f"Sampled {n_samples}/{total_size} samples (frac={sample_frac})")

        return Subset(dataset, indices)
# endregion


# region Main Inference Runner
def run_inference(
    output_dir: str,
    checkpoint_path: str,
    hparams: Dict[str, Any],
    data_path: str,
    batch_size: int = 32,
    save_plot: bool = False,
    conversion_dict_path: Optional[str] = None,
    sample_frac: float = 1.0,
    data_dir: str = ".",
    output_format: str = "h5",
    plot_limit: int = 10,
    plot_original: bool = False,
    data_pair_path:str=None,
    n_jobs_io: int = 8,
    sample_seed: int = 42,
    is_pair: bool = False,
    mode: Optional[str] = None,
    show_progressbar: bool = True,
    indices_path: Optional[str] = None,
    **kwargs
) -> None:
    """Run inference on a trained model.

    The function loads the model and its transformation pipelines, prepares the dataset
    using the input-domain transformers, runs the model, inverts output-domain transforms
    and writes results with the requested writers. TXT files contain two columns: q and y.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running inference on device: {device}")

    logger.info(f"Loading model from: {checkpoint_path}")
    model = ModelLoader.load_model(checkpoint_path, hparams, device)

    # Prepare transformers depending on model type and mode
    if mode is None:
        try:
            transforms_data = model.get_transformer()
        except Exception as e:
            raise RuntimeError(f"Unable to obtain transforms from model: {e}")

        try:
            model_spec = model.model_cfg.spec
        except Exception:
            model_spec = ModelSpec.SAXS

        inference_mode = f"recon_{model_spec.value}"
        use_loglog = (model_spec == ModelSpec.SAXS)

        if 'q' not in transforms_data or 'y' not in transforms_data:
            raise RuntimeError("Model transforms_data must include 'q' and 'y' entries")

        transformer_q_input = Pipeline(transforms_data['q'])
        transformer_y_input = Pipeline(transforms_data['y'])
        transformer_q_output = Pipeline(transforms_data['q'])
        transformer_y_output = Pipeline(transforms_data['y'])

        is_pair_model = False

    else:
        mode_map = {
            'les_to_saxs': ('les', 'saxs'),
            'saxs_to_les': ('saxs', 'les'),
            'les_to_les': ('les', 'les'),
            'saxs_to_saxs': ('saxs', 'saxs'),
        }

        if mode not in mode_map:
            raise ValueError(f"Invalid PairVAE mode: {mode}. Must be one of {list(mode_map.keys())}")

        input_domain, output_domain = mode_map[mode]
        use_loglog = (output_domain == 'saxs')
        inference_mode = mode

        try:
            if input_domain == 'saxs':
                transforms_input = model.get_transforms_data_saxs()
            else:
                transforms_input = model.get_transforms_data_les()

            if output_domain == 'saxs':
                transforms_output = model.get_transforms_data_saxs()
            else:
                transforms_output = model.get_transforms_data_les()
        except Exception as e:
            raise RuntimeError(f"Unable to obtain pair transforms from model: {e}")

        if 'q' not in transforms_input or 'y' not in transforms_input:
            raise RuntimeError("Input transforms must include 'q' and 'y'")
        if 'q' not in transforms_output or 'y' not in transforms_output:
            raise RuntimeError("Output transforms must include 'q' and 'y'")

        transformer_q_input = Pipeline(transforms_input['q'])
        transformer_y_input = Pipeline(transforms_input['y'])
        transformer_q_output = Pipeline(transforms_output['q'])
        transformer_y_output = Pipeline(transforms_output['y'])

        is_pair_model = True
        model_spec = ModelSpec(output_domain)

    logger.info(f"Loading data from: {data_path}")
    dataset = DatasetLoader.load_dataset(
        data_path=data_path,
        conversion_dict_path=conversion_dict_path,
        transformer_q=transformer_q_input,
        transformer_y=transformer_y_input,
        data_dir=data_dir,
        show_progressbar=show_progressbar,
    )

    dataset = DatasetSampler.sample_dataset(dataset, sample_frac, sample_seed, indices_path, model_spec)
    dataloader = DatasetLoader.prepare_dataloader(dataset, batch_size)

    logger.info(f"Running inference... with output spec: {model_spec.value}, ")
    engine = InferenceEngine(model, transformer_y_output, transformer_q_output, device)

    if is_pair_model:
        results = engine.run_pairvae_inference(dataloader, mode)
    else:
        results = engine.run_vae_inference(dataloader)

    logger.info(f"Inference complete. Processed {len(results.get('predictions', []))} samples.")

    writers: List[OutputWriter] = []

    if output_format == 'h5':
        output_path = str(Path(output_dir) / f"predictions_{inference_mode}.h5")
        writers.append(HDF5Writer(output_path))
    elif output_format == 'txt':
        output_path = str(Path(output_dir) / f"predictions_{inference_mode}")
        writers.append(TXTWriter(output_path, n_jobs=n_jobs_io))
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    if save_plot:
        plot_dir = str(Path(output_dir) / f"plots_{inference_mode}")
        writers.append(PlotWriter(plot_dir, plot_limit=plot_limit, use_loglog=use_loglog, plot_original=plot_original, output_domain=output_domain, data_pair_path=data_pair_path, is_pair=is_pair, input_domain=input_domain))

    for writer in writers:
        writer.write(results)

    logger.info("Inference pipeline completed successfully!")
# endregion
