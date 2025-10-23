"""PyTorch dataset for single-spectrum HDF5 files."""

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
from src.logging_utils import get_logger


logger = get_logger(__name__)

DEFAULT_LENGTH_DATA = 1000


class HDF5Dataset(Dataset):
    """Dataset loader for single-spectrum experiments stored in HDF5 files."""

    def __init__(self, hdf5_file, conversion_dict: Union[dict, str, Path] = None, metadata_filters=None,
                 sample_frac=1, requested_metadata=None,
                 transformer_q=Pipeline(), transformer_y=Pipeline(),
                 use_data_q: bool = True):
        """
        Initialize the dataset and eagerly prepare metadata filters and transforms.

        Args:
            hdf5_file (str | Path): Path to the HDF5 file.
            conversion_dict (dict | str | Path, optional): Conversion dictionary for categorical metadata.
            metadata_filters (dict, optional): Metadata filters to restrict samples.
            sample_frac (float, optional): Fraction of samples to load (0 < sample_frac <= 1).
            requested_metadata (list, optional): Metadata columns to include.
            transformer_q (Pipeline): Transformation pipeline for q values.
            transformer_y (Pipeline): Transformation pipeline for y values.
            use_data_q (bool): Whether to load and include data_q in batches. If False, returns None for data_q.
        """
        if requested_metadata is None:
            requested_metadata = []

        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(hdf5_file, 'r', swmr=True)
        self.use_data_q = use_data_q

        # --- Load q or wavelength data ---
        if self.use_data_q:
            try:
                if "data_q" in self.hdf:
                    self.data_q = self.hdf["data_q"]
                else:
                    self.data_q = self.hdf["data_wavelength"]
            except Exception as e:
                raise ValueError(f"Error loading 'data_q' or 'data_wavelength' from HDF5 file: {e}")

            # Sanity check: ensure all q arrays are identical
            first_q = self.data_q[0]
            for i in tqdm(range(len(self.data_q)), desc="Sanity checking H5", leave=False):
                if not np.array_equal(self.data_q[i], first_q):
                    raise AssertionError("All data_q/data_wavelength arrays must be identical")
            self.data_q = first_q
        else:
            self.data_q = None

        # --- Load intensity data ---
        self.data_y = self.hdf['data_y']
        assert len(self.data_y) > 0 or (self.data_q is not None and len(self.data_q) > 0), (
            "H5 file is empty. Check your metadata filters and make sure they are not too restrictive."
        )

        self.transformer_q = _ensure_pipeline(transformer_q)
        self.transformer_y = _ensure_pipeline(transformer_y)

        self.csv_index = self.hdf.get('csv_index', None)
        if "len" in self.hdf.keys():
            self.len = self.hdf["len"][()]
        else:
            self.len = np.full(len(self.data_y), DEFAULT_LENGTH_DATA)

        all_metadata_cols = [col for col in self.hdf.keys() if col not in
                             ['data_q', 'data_y', 'len', 'data_wavelength']]
        self.metadata_datasets = {col: self.hdf[col] for col in all_metadata_cols}
        self.requested_metadata = self._validate_requested_metadata(requested_metadata, all_metadata_cols)

        # --- Conversion dict and filtering ---
        if conversion_dict is not None:
            self.conversion_dict = self._load_conversion_dict(conversion_dict)
            self.metadata_filters = metadata_filters or {}
            self.filtered_indices = self._apply_metadata_filters()
        else:
            self.conversion_dict = None
            self.filtered_indices = list(range(len(self.data_y)))

        # --- Sampling fraction ---
        self._validate_frac(sample_frac)
        self.sample_frac = sample_frac
        if 0 < sample_frac < 1:
            self._apply_data_fraction(sample_frac)

        self._print_init_info()

        # --- Fit transformers ---
        self.transformer_y.fit(self.data_y[self.filtered_indices])
        if self.use_data_q and self.data_q is not None:
            self.transformer_q.fit(self.data_q)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def _print_init_info(self):
        """Display dataset loading summary."""
        logger.info(
            "Loaded %s with %d filtered samples out of %d",
            self.hdf5_file,
            len(self.filtered_indices),
            len(self.data_y),
        )

    def _validate_requested_metadata(self, requested, available):
        """Validate and filter requested metadata columns."""
        if requested is None:
            return available

        valid = [col for col in requested if col in available]
        missing = set(requested) - set(valid)
        if missing:
            warnings.warn(f"Missing requested metadata columns: {missing}, available: {available}")
        return valid

    def _validate_frac(self, sample_frac):
        """Validate that sample fraction is in the (0,1] interval."""
        if not (0 < sample_frac <= 1):
            raise ValueError("Data fraction must be between 0 and 1")

    def _load_conversion_dict(self, conversion_dict: Union[dict, str, Path]):
        """Load JSON conversion dictionary for categorical metadata."""
        if isinstance(conversion_dict, (str, Path)):
            with open(conversion_dict, 'r') as f:
                conversion_dict = json.load(f)
        elif not isinstance(conversion_dict, dict):
            raise ValueError("Conversion dictionary must be a dict or path to JSON file")
        return conversion_dict

    def _apply_metadata_filters(self):
        """Vectorized metadata filtering using numpy operations."""
        if not self.metadata_filters:
            return list(range(len(self.data_y)))

        mask = np.ones(len(self.data_y), dtype=bool)
        for key, allowed_values in tqdm(self.metadata_filters.items(), desc="Applying filters"):
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
        """Select a fraction of the filtered indices in sorted order."""
        num_samples = int(len(self.filtered_indices) * sample_frac)
        self.filtered_indices = self.filtered_indices[:num_samples]

    # -------------------------------------------------------------------------
    # PyTorch dataset interface
    # -------------------------------------------------------------------------

    def __len__(self):
        """Return number of filtered samples available in the dataset."""
        return len(self.filtered_indices)

    def _get_metadata(self, idx):
        """Retrieve requested metadata for a given sample."""
        metadata = {}
        for col in self.requested_metadata:
            data = self.metadata_datasets[col][idx]
            metadata[col] = data
        return metadata

    def __getitem__(self, idx):
        """Return the transformed tensors and metadata for the requested sample."""
        original_idx = self.filtered_indices[idx]
        data_y = self.data_y[original_idx]

        metadata = {k: torch.tensor(self.metadata_datasets[k][original_idx])
                    for k in self.requested_metadata}

        data_y_transformed = self.transformer_y.transform(data_y)
        data_y_transformed = torch.as_tensor(data_y_transformed, dtype=torch.float32).unsqueeze(0)

        if self.use_data_q and self.data_q is not None:
            data_q = self.transformer_q.transform(self.data_q)
            data_q = torch.as_tensor(data_q, dtype=torch.float32).unsqueeze(0)
        else:
            data_q = None

        batch = {
            "data_y": data_y_transformed,
            "data_y_untransformed": torch.as_tensor(data_y, dtype=torch.float32).unsqueeze(0),
            "metadata": metadata,
            "len": self.len[original_idx],
        }

        if self.csv_index is not None:
            batch["csv_index"] = self.csv_index[original_idx]

        if data_q is not None:
            batch["data_q"] = data_q

        return batch

    # -------------------------------------------------------------------------
    # Convenience and export helpers
    # -------------------------------------------------------------------------

    def close(self):
        """Close the underlying HDF5 file."""
        self.hdf.close()

    def get_conversion_dict(self):
        """Return the conversion dictionary for categorical metadata."""
        return self.conversion_dict

    def transforms_to_dict(self):
        """Convert transformation pipelines to a dictionary format for saving."""
        return {
            "q": self.transformer_q.to_dict(),
            "y": self.transformer_y.to_dict()
        }

    def get_data_q(self):
        """Return the original data_q array (if loaded)."""
        return self.data_q

    def invert_transforms_func(self):
        """Return the inverse transformation functions for data_y and data_q."""
        def func(y_arr, q_arr):
            y_arr = self.transformer_y.invert(y_arr)
            q_arr = self.transformer_q.invert(q_arr) if self.use_data_q and q_arr is not None else None
            return y_arr, q_arr
        return func


def _ensure_pipeline(transformer) -> Pipeline:
    """Ensure transformer is a valid Pipeline instance."""
    if isinstance(transformer, Pipeline):
        return transformer
    try:
        return Pipeline(transformer)
    except Exception as e:
        raise ValueError(f"Invalid transformer config: {e}")
