import json
import os
import warnings
from pathlib import Path
import numpy as np

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset.transformations import *

DEFAULT_LENGTH_DATA = 1000

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, conversion_dict: Union[dict, str, Path]=None, metadata_filters=None,
                 sample_frac=1, requested_metadata=[],
                 transformer_q=Pipeline(), transformer_y=Pipeline()):
        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(hdf5_file, 'r', swmr=True)
        try:
            if "data_q" in self.hdf:    
                self.data_q = self.hdf["data_q"]
            else:
                self.data_q = self.hdf["data_wavelength"]    
        except Exception as e:
            raise ValueError(f"Error loading 'data_q' or 'data_wavelength' from HDF5 file: {e}")

        #indices = np.where(self.data_q[0] > 0.3)[0]
        #print(indices)

        self.data_y = self.hdf['data_y']
        assert len(self.data_q) == len(self.data_y), "data_q and data_y must have the same length"
        assert len(self.data_y) > 0 or len(self.data_q) >0, "H5file is empty, please check your HDF5 file\n" \
        "Check your metadata filters and make sure they are not too restrictive."

        self.transformer_q = _ensure_pipeline(transformer_q)
        self.transformer_y = _ensure_pipeline(transformer_y)

        self.csv_index = self.hdf['csv_index']
        if "len" in self.hdf.keys():
            self.len = self.hdf["len"][()]  # lire la valeur du dataset
        else:
            self.len = arr = np.full(len(self.hdf['data_y']), DEFAULT_LENGTH_DATA)

        all_metadata_cols = [col for col in self.hdf.keys() if col not in
                             ['data_q', 'data_y', 'len', 'csv_index']]
        self.metadata_datasets = {col: self.hdf[col] for col in all_metadata_cols}
        self.requested_metadata = self._validate_requested_metadata(requested_metadata, all_metadata_cols)

        if conversion_dict is not None :
            self.conversion_dict = self._load_conversion_dict(conversion_dict)
            self.metadata_filters = metadata_filters or {}
            self.filtered_indices = self._apply_metadata_filters()

        else :
            self.conversion_dict = None
            self.filtered_indices = list(range(len(self.data_q)))

        self._validate_frac(sample_frac)
        self.sample_frac = sample_frac
        if 0 < sample_frac < 1:
            self._apply_data_fraction(sample_frac)

        self._print_init_info()
        
        self.transformer_y.fit(self.data_y[self.filtered_indices])
        self.transformer_q.fit(self.data_q[self.filtered_indices])

    def _print_init_info(self):
        print(
            f"[HDF5Dataset] File: {self.hdf5_file} | Filtered samples: {len(self.filtered_indices)} / {len(self.data_q)}")

    def _validate_requested_metadata(self, requested, available):
        """Validate and filter requested metadata columns"""
        if requested is None:
            return available

        valid = [col for col in requested if col in available]
        missing = set(requested) - set(valid)

        if missing:
            warnings.warn(f"Missing requested metadata columns: {missing}, available: {available}")

        return valid

    def _validate_frac(self, sample_frac):
        """Validate data fraction parameter"""
        if not (0 < sample_frac <= 1):
            raise ValueError("Data fraction must be between 0 and 1")

    def _load_conversion_dict(self, conversion_dict: Union[dict, str, Path]):
        """Load JSON conversion dictionary for categorical metadata"""
        if isinstance(conversion_dict, (str, Path)):
            with open(conversion_dict, 'r') as f:
                conversion_dict = json.load(f)
        elif not isinstance(conversion_dict, dict):
            raise ValueError("Conversion dictionary must be a dictionary or a path to a JSON file")

        return conversion_dict

    def _apply_metadata_filters(self):
        """Vectorized metadata filtering using numpy operations"""
        if not self.metadata_filters:
            return list(range(len(self.data_q)))

        mask = np.ones(len(self.data_q), dtype=bool)

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
        return len(self.filtered_indices)

    def _get_metadata(self, idx):
        """Preprocess requested metadata to tensors during initialization"""
        metadata = {}
        for col in self.requested_metadata:
            data = self.metadata_datasets[col][idx]
            metadata[col] = data
        return metadata

    def get_conversion_dict(self):
        """Return the conversion dictionary for categorical metadata"""
        return self.conversion_dict

    def __getitem__(self, idx):
        """Optimized data loading with timing and progress tracking"""
        # Get original dataset index
        original_idx = self.filtered_indices[idx]

        # Load main data
        data_q = self.data_q[original_idx]
        data_y = self.data_y[original_idx]

        # Get preprocessed metadata
        metadata = self._get_metadata(original_idx)
        metadata = {k: torch.tensor(v) for k, v in metadata.items()}

        # Data processing
        data_q = self.transformer_q.transform(data_q)
        data_y = self.transformer_y.transform(data_y)

        # Convert to tensors if needed
        data_q = torch.as_tensor(data_q, dtype=torch.float32)
        data_y = torch.as_tensor(data_y, dtype=torch.float32)

        # return data_q, data_y, metadata, self.csv_index[original_idx]
        return {"data_q": data_q.unsqueeze(0), "data_y": data_y.unsqueeze(0),
                "metadata": metadata, "csv_index": self.csv_index[original_idx], "len": self.len[original_idx]}

    def close(self):
        """Close the HDF5 file"""
        self.hdf.close()

    def transforms_to_dict(self):
        """Convert the transformations to a dictionary format for saving"""
        return {
            "q": self.transformer_q.to_dict(),
            "y": self.transformer_y.to_dict()
        }

    def invert_transforms_func(self):
        """Return the inverse transformation functions for data_y and data_q"""
        def func(y_arr, q_arr):
            y_arr = self.transformer_y.invert(y_arr)
            q_arr = self.transformer_q.invert(q_arr)
            return y_arr, q_arr
        return func

def _ensure_pipeline(transformer) -> Pipeline:
    if isinstance(transformer, Pipeline):
        return transformer
    try:
        return Pipeline(transformer)
    except Exception as e:
        raise ValueError(f"Invalid {transformer}: {e}")