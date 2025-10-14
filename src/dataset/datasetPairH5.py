"""PyTorch dataset for paired SAXS and LES spectra stored in HDF5 files."""

import json
import warnings
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset.transformations import Pipeline


class PairHDF5Dataset(Dataset):
    """Dataset wrapper for loading coupled SAXS/LES experiments."""

    def __init__(self, hdf5_file, conversion_dict: Union[dict, str, Path], metadata_filters=None,
                 sample_frac=1, requested_metadata=None,
                 transformer_q_saxs=Pipeline(),
                 transformer_y_saxs=Pipeline(),
                 transformer_q_les=Pipeline(),
                 transformer_y_les=Pipeline(),
                 **kwargs):
        """Initialize paired datasets and configure metadata filtering and transforms."""
        if requested_metadata is None:
            requested_metadata = []
        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(hdf5_file, 'r', swmr=True)

        required_keys = ['data_q_saxs', 'data_y_saxs', 'data_q_les', 'data_y_les', 'csv_index']
        missing = [k for k in required_keys if k not in self.hdf]
        if missing:
            raise RuntimeError(
                f"Missing required datasets in HDF5 file: {missing}\n"
                "Your HDF5 file is not compatible with PairVAE. "
                "Refer to the README (section 5) and generate it using scripts/05_pair_txtTOhdf5.py."
            )

        self.data_q_saxs = self.hdf['data_q_saxs']
        self.data_y_saxs = self.hdf['data_y_saxs']
        self.data_q_les = self.hdf['data_q_les']
        self.data_y_les = self.hdf['data_y_les']
        self.csv_index = self.hdf['csv_index']


        assert all(self.data_q_saxs.apply(lambda x: np.array_equal(x, self.data_q_saxs[0]))), "All data_q_saxs arrays must be identical"
        assert all(self.data_q_les.apply(lambda x: np.array_equal(x, self.data_q_les[0]))), "All data_wavelength arrays must be identical"
        assert len(self.data_q_saxs) == len(self.data_y_saxs), "data_y_saxs and data_q_saxs must have the same length"
        assert len(self.data_q_les) == len(self.data_y_les), "data_y_saxs and data_q_saxs must have the same length"
        assert len(self.data_q_saxs) > 0 or len(self.data_y_saxs) > 0, (
            "H5file saxs part is empty, please check your HDF5 file\n"
            "Check your metadata filters and make sure they are not too restrictive."
        )
        assert len(self.data_q_les) > 0 or len(self.data_y_les) > 0, (
            "H5file les part is empty, please check your HDF5 file\n"
            "Check your metadata filters and make sure they are not too restrictive."
        )

        self.transformer_q_saxs = _ensure_pipeline(transformer_q_saxs)
        self.transformer_y_saxs = _ensure_pipeline(transformer_y_saxs)
        self.transformer_q_les = _ensure_pipeline(transformer_q_les)
        self.transformer_y_les = _ensure_pipeline(transformer_y_les)

        all_metadata_cols = [col for col in self.hdf.keys() if col not in
                             ['data_q_saxs', 'data_y_saxs', 'data_q_les', 'data_y_les', 'len', 'csv_index']]
        self.metadata_datasets = {col: self.hdf[col] for col in all_metadata_cols}

        self.requested_metadata = self._validate_requested_metadata(requested_metadata, all_metadata_cols)
        if conversion_dict is not None :
            self.conversion_dict = self._load_conversion_dict(conversion_dict)
            self.metadata_filters = metadata_filters or {}
            self.filtered_indices = self._apply_metadata_filters()
        else :
            self.conversion_dict = None
            self.filtered_indices = list(range(len(self.data_q_les)))

        self.metadata_filters = metadata_filters or {}
        self.filtered_indices = self._apply_metadata_filters()

        self._validate_frac(sample_frac)
        self.sample_frac = sample_frac
        if 0 < sample_frac < 1:
            self._apply_data_fraction(sample_frac)

        self._print_init_info()

        self.transformer_y_saxs.fit(self.data_y_saxs[self.filtered_indices])
        self.transformer_y_les.fit(self.data_y_les[self.filtered_indices])

    def _print_init_info(self):
        """Print dataset initialization information"""
        print("\n╒══════════════════════════════════════════════╕")
        print("│ Dataset Initialization Info                 │")
        print("╞══════════════════════════════════════════════╡")
        print(f"│ File: {self.hdf5_file:<35} │")
        print(f"│ Total samples: {len(self.data_q_saxs):<26} │")
        print(f"│ Samples filtered: {len(self.filtered_indices):<23} │")
        print(f"│ Requested fraction: {self.sample_frac:<22} │")
        print(f"│ Fractioned samples: {len(self.filtered_indices):<22} │")
        print(f"│ Requested metadata: {len(self.requested_metadata):<22} │")

    def _validate_requested_metadata(self, requested, available):
        """Validate and filter requested metadata columns"""
        if requested is None:
            return available

        valid = [col for col in requested if col in available]
        missing = set(requested) - set(valid)

        if missing:
            warnings.warn(f"Missing requested metadata columns: {missing}")

        return valid

    def _validate_frac(self, sample_frac):
        """Validate data fraction parameter"""
        if not (0 < sample_frac <= 1):
            raise ValueError("Data fraction must be between 0 and 1")

    def _load_conversion_dict(self, conversion_dict: Union[dict, str, Path]):
        """Load JSON conversion dictionary for categorical metadata"""
        assert conversion_dict is not None, "A conversion dictionary must be provided"
        if isinstance(conversion_dict, (str, Path)):
            with open(conversion_dict, 'r') as f:
                conversion_dict = json.load(f)
        elif not isinstance(conversion_dict, dict):
            raise ValueError("Conversion dictionary must be a dictionary or a path to a JSON file")
        return conversion_dict

    def get_conversion_dict(self):
        """Return the conversion dictionary for categorical metadata"""
        return self.conversion_dict

    def _apply_metadata_filters(self):
        """Vectorized metadata filtering using numpy operations"""
        if not self.metadata_filters:
            return list(range(len(self.data_q_saxs)))

        mask = np.ones(len(self.data_q_saxs), dtype=bool)

        for key, allowed_values in tqdm(self.metadata_filters.items(),
                                        desc="Applying filters"):
            if key not in self.metadata_datasets:
                mask[:] = False
                break

            data = self.metadata_datasets[key][...]

            if key in self.conversion_dict:
                converted_allowed = [self.conversion_dict[key].get(str(v), -1)
                                     for v in allowed_values]
                key_mask = np.isin(data, converted_allowed)
            else:
                key_mask = np.isin(data, allowed_values)

            mask &= key_mask

        return np.where(mask)[0]

    def _apply_data_fraction(self, sample_frac):
        """Select a fraction of the filtered indices in sorted order"""
        num_samples = int(len(self.filtered_indices) * sample_frac)
        self.filtered_indices = self.filtered_indices[:num_samples]

    def __len__(self):
        """Return the number of filtered SAXS/LES sample pairs."""
        return len(self.filtered_indices)

    def _get_metadata(self, idx):
        """Preprocess requested metadata to tensors during initialization"""
        metadata = {}
        for col in self.requested_metadata:
            data = self.metadata_datasets[col][idx]
            metadata[col] = data
        return metadata

    def __getitem__(self, idx):
        """Return the paired SAXS/LES tensors and metadata for a sample."""
        original_idx = self.filtered_indices[idx]

        data_q_saxs = self.data_q_saxs[original_idx]
        data_y_saxs = self.data_y_saxs[original_idx]
        data_q_les = self.data_q_les[original_idx]
        data_y_les = self.data_y_les[original_idx]

        metadata = self._get_metadata(original_idx)
        metadata = {k: torch.tensor(v) for k, v in metadata.items()}

        data_q_saxs = self.transformer_q_saxs.transform(data_q_saxs)
        data_y_saxs = self.transformer_y_saxs.transform(data_y_saxs)
        data_q_les = self.transformer_q_les.transform(data_q_les)
        data_y_les = self.transformer_y_les.transform(data_y_les)

        data_q_saxs = torch.as_tensor(data_q_saxs, dtype=torch.float32)
        data_y_saxs = torch.as_tensor(data_y_saxs, dtype=torch.float32)
        data_q_les = torch.as_tensor(data_q_les, dtype=torch.float32)
        data_y_les = torch.as_tensor(data_y_les, dtype=torch.float32)

        for name, arr in zip([
            'data_q_saxs', 'data_y_saxs', 'data_q_les', 'data_y_les'],
            [data_q_saxs, data_y_saxs, data_q_les, data_y_les]):
            if torch.isnan(arr).any() or torch.isinf(arr).any():
                raise RuntimeError(f"[PairHDF5Dataset][idx={idx}] {name} contient NaN ou inf!")

        return {"data_q_saxs": data_q_saxs.unsqueeze(0), "data_y_saxs": data_y_saxs.unsqueeze(0),
            "data_q_les": data_q_les.unsqueeze(0), "data_y_les": data_y_les.unsqueeze(0),
            "metadata": metadata, "csv_index": self.csv_index[original_idx]}

    def close(self):
        """Close the HDF5 file"""
        self.hdf.close()

    def transforms_to_dict(self):
        """Convert the transformations to a dictionary format"""
        return {
            "q_saxs": self.transformer_q_saxs.to_dict(),
            "y_saxs": self.transformer_y_saxs.to_dict(),
            "q_les": self.transformer_q_les.to_dict(),
            "y_les": self.transformer_y_les.to_dict()
        }

    def get_data_q_saxs(self):
        """Return the original data_q_saxs array"""
        return self.data_q_saxs

    def get_data_q_les(self):
        """Return the original data_q_les array"""
        return self.data_q_les


def _ensure_pipeline(transformer) -> Pipeline:
    """Ensure that a transformer configuration is represented as a :class:`Pipeline`."""
    if isinstance(transformer, Pipeline):
        return transformer
    try:
        return Pipeline(transformer)
    except Exception as e:
        raise ValueError(f"Invalid {transformer}: {e}")