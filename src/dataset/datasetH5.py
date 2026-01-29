"""PyTorch dataset for single-spectrum HDF5 files."""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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

    def __init__(self, hdf5_file, conversion_dict: Union[dict, str, Path] = None, metadata_filters=None, requested_metadata=None,
                 transformer_q=Pipeline(), transformer_y=Pipeline(),
                 use_data_q: bool = True, sanity_check=True, show_progressbar: bool = True):
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
        self.show_progress = show_progressbar
        self.hdf5_file = Path(hdf5_file)
        #self.hdf = h5py.File(hdf5_file, 'r', swmr=True)
        self._hdf_handle = None
        self.use_data_q = use_data_q
        self.sanity_check = sanity_check

        with h5py.File(self.hdf5_file, 'r') as hdf:
            q_key = "data_q" if "data_q" in hdf else "data_wavelength"
            if q_key not in hdf:
                raise ValueError(f"Could not find 'data_q' or 'data_wavelength' in {hdf5_file}")

            temp_data_q = hdf[q_key]
            first_q = temp_data_q[0]
            
            if self.sanity_check:
                for i in tqdm(range(len(temp_data_q)), desc="Sanity checking H5", leave=False, disable=not self.show_progress, mininterval=1, miniters=min(10_000, max(1, len(temp_data_q) // 100))):
                    if not np.array_equal(temp_data_q[i], first_q):
                        raise AssertionError("All data_q/data_wavelength arrays must be identical")
            self.data_q = first_q

            # --- Load intensity data ---

            assert len(hdf['data_y']) > 0 or (temp_data_q is not None and len(temp_data_q) > 0), (
                "H5 file is empty. Check your metadata filters and make sure they are not too restrictive."
            )

            num_samples = len(hdf['data_y'])
            if "len" in hdf:
                self.len_attr = hdf["len"][()]
            else:
                self.len_attr = np.full(num_samples, DEFAULT_LENGTH_DATA)

            self.available_metadata = [col for col in hdf.keys() if col not in
                                     ['data_q', 'data_y', 'len', 'data_wavelength']]
            self.requested_metadata = self._validate_requested_metadata(requested_metadata, self.available_metadata)

            # --- Conversion dict and filtering ---
            if conversion_dict is not None:
                self.conversion_dict = self._load_conversion_dict(conversion_dict)
                self.metadata_filters = metadata_filters or {}
                self.filtered_indices = self._apply_metadata_filters(hdf)
            else:
                self.conversion_dict = None
                self.filtered_indices = np.arange(num_samples)

            self._print_init_info()

            # --- Fit transformers ---
            self.transformer_q = _ensure_pipeline(transformer_q)
            self.transformer_y = _ensure_pipeline(transformer_y)
            
            self.transformer_y.fit(hdf['data_y'][self.filtered_indices])
            if self.use_data_q:
                self.transformer_q.fit(self.data_q)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    @property
    def hdf(self):
        """Lazy loader for the HDF5 file handle."""
        if self._hdf_handle is None:
            self._hdf_handle = h5py.File(self.hdf5_file, 'r', swmr=True, libver='latest')
        return self._hdf_handle

    def _print_init_info(self):
        """Display dataset loading summary."""
        logger.info("Loaded %s with %d samples", self.hdf5_file.name, len(self.filtered_indices))

    def _validate_requested_metadata(self, requested, available):
        """Validate and filter requested metadata columns."""
        if not requested: return available
        valid = [col for col in requested if col in available]
        missing = set(requested) - set(valid)
        if missing:
            logger.warning(f"Missing metadata: {missing}")
        return valid

    def _load_conversion_dict(self, conversion_dict: Union[dict, str, Path]):
        """Load JSON conversion dictionary for categorical metadata."""
        if isinstance(conversion_dict, (str, Path)):
            with open(conversion_dict, 'r') as f:
                return json.load(f)
        return conversion_dict

    def _apply_metadata_filters(self, hdf_handle):
        """Vectorized metadata filtering using numpy operations."""
        if not self.metadata_filters:
            return np.arange(len(hdf_handle['data_y']))

        mask = np.ones(len(hdf_handle['data_y']), dtype=bool)
        for key, allowed_values in tqdm(self.metadata_filters.items(), desc="Applying filters", disable=not self.show_progress):
            if key not in hdf_handle:
                mask[:] = False
                break

            data = hdf_handle[key][...]
            if key in self.conversion_dict:
                conv = self.conversion_dict[key]
                converted_allowed = [conv.get(str(v), -1) for v in allowed_values]
                key_mask = np.isin(data, converted_allowed)
            else:
                key_mask = np.isin(data, allowed_values)
            mask &= key_mask
        return np.where(mask)[0]

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
        data_y = self.hdf['data_y'][original_idx]

        metadata = {k: torch.tensor(self.hdf[k][original_idx])
                    for k in self.requested_metadata}

        data_y_transformed = self.transformer_y.transform(data_y)
        data_y_transformed = torch.as_tensor(data_y_transformed, dtype=torch.float32).unsqueeze(0)

        data_q = self.transformer_q.transform(self.data_q)
        data_q = torch.as_tensor(data_q, dtype=torch.float32).unsqueeze(0)

        batch = {"data_y": data_y_transformed,
                 "data_y_untransformed": torch.as_tensor(data_y, dtype=torch.float32).unsqueeze(0),
                 "metadata": metadata, "len": self.len_attr[original_idx], "data_index": original_idx, "data_q": data_q}

        return batch
    
    def __getstate__(self):
        """Standard pickle logic: remove the file handle before serializing."""
        state = self.__dict__.copy()
        state['_hdf_handle'] = None
        return state
    
    def __setstate__(self, state):
        """Standard pickle logic: restore state, handle will reopen lazily."""
        self.__dict__.update(state)
        self._hdf_handle = None

    # -------------------------------------------------------------------------
    # Convenience and export helpers
    # -------------------------------------------------------------------------

    def close(self):
        """Close the underlying HDF5 file."""
        if self._hdf_handle is not None:
            self._hdf_handle.close()
            self._hdf_handle = None

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
        return np.array(self.data_q)

    def invert_transforms_func(self):
        """Return the inverse transformation functions for data_y and data_q."""
        def func(y_arr, q_arr):
            y_arr = self.transformer_y.invert(y_arr)
            q_arr = self.transformer_q.invert(q_arr)
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
