"""PyTorch dataset helper for spectroscopy data stored as TXT files."""

from pathlib import Path
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset.transformations import Pipeline
from src.logging_utils import get_logger


logger = get_logger(__name__)


class TXTDataset(Dataset):
    """Dataset for spectra stored as TXT/CSV files with cached preprocessing."""

    def __init__(self, dataframe, data_dir=Path('./'),
                 transformer_q=Pipeline(), transformer_y=Pipeline(), cache_limit=250000):
        """Create a dataset backed by an index dataframe and lazily loaded files."""
        self.dataframe = dataframe
        self.categorical_cols = ['material', 'type', 'method', 'shape', 'researcher', 'technique']
        self.numerical_cols = ['concentration', 'opticalPathLength', 'd', 'h']
        self.data_dir = Path(data_dir)
        self.cache_limit = cache_limit
        self.transformer_q = _ensure_pipeline(transformer_q)
        self.transformer_y = _ensure_pipeline(transformer_y)
        self.fit_transformers()
        self.cat_vocab = self._build_cat_vocab()
        self.metadata_tensor = self._preprocess_metadata()
        self.data_cache = OrderedDict()
        self.sample_count = 0

    def __len__(self):
        """Return the number of rows in the metadata dataframe."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Return tensors for the spectrum referenced by ``idx`` and cached metadata."""
        metadata = self.metadata_tensor[idx]
        relative_path = self.dataframe.iloc[idx]['path']
        file_path = self.data_dir / relative_path

        if file_path in self.data_cache:
            data_q, data_y = self.data_cache.pop(file_path)
            self.data_cache[file_path] = (data_q, data_y)
        else:
            try:
                q, y = self._load_data_from_file(file_path)
            except Exception as e:
                warnings.warn(f"Erreur fichier {file_path}: {e}")
                q = torch.zeros(1)
                y = torch.zeros(1)
            self.data_cache[file_path] = (q, y)
            if len(self.data_cache) > self.cache_limit:
                self.data_cache.popitem(last=False)

        data_q, data_y = self.data_cache[file_path]
        data_y_min = data_y.min()
        data_y_max = data_y.max()
        data_q = self.transformer_q.transform(data_q)
        data_y = self.transformer_y.transform(data_y)
        data_q = torch.as_tensor(data_q, dtype=torch.float32)
        data_y = torch.as_tensor(data_y, dtype=torch.float32)
        return {
            "data_q": data_q.unsqueeze(0),
            "data_y": data_y.unsqueeze(0),
            "data_y_min": data_y_min,
            "data_y_max": data_y_max,
            "metadata": metadata,
            "data_index": idx,
            "path": str(file_path)
        }

    def fit_transformers(self):
        """Fit the preprocessing pipelines on the referenced dataset."""
        q_data= []
        y_data = []
        if self.transformer_q.is_fitted() and self.transformer_y.is_fitted():
            logger.info("Transformers already fitted; skipping fitting")
            return
        for file_path in tqdm(self.dataframe['path'], desc="Fitting transformers"):
            file_path = self.data_dir / file_path
            try:
                q, y = self._load_data_from_file(file_path)
                q_data.append(q)
                y_data.append(y)
            except Exception as e:
                warnings.warn(f"Erreur fichier {file_path}: {e}")
        self.transformer_q.fit(q_data)
        self.transformer_y.fit(y_data)


    def _build_cat_vocab(self):
        """Return categorical vocabularies for each metadata column."""
        return {
            col: {val: idx for idx, val in enumerate(self.dataframe[col].astype(str).unique())}
            for col in self.categorical_cols
        }

    def _preprocess_metadata(self):
        """Convert categorical and numerical metadata columns into tensors."""
        cat_data = [
            self.dataframe[col].astype(str).map(self.cat_vocab[col]).fillna(-1).astype(float)
            for col in self.categorical_cols
        ]
        num_data = self.dataframe[self.numerical_cols].fillna(0.0).astype(float)
        combined = pd.concat([pd.DataFrame(cat_data).T, num_data], axis=1)
        return torch.tensor(combined.values, dtype=torch.float32)

    def _load_data_from_file(self, file_path: Path):
        """Load a single TXT file as ``(q, y)`` tensors."""
        try:
            df = pd.read_csv(
                file_path,
                comment='#',
                sep='[;,\\s]+',
                engine='python',
                usecols=[0, 1],
                names=['q', 'y'],
                header=0,
                dtype=np.float32
            )
        except Exception as e:
            raise ValueError(f"Read error ({file_path}): {e}")
        df = df.dropna().replace([np.inf, -np.inf], 0.0)
        return (
            torch.tensor(df['q'].values, dtype=torch.float32),
            torch.tensor(df['y'].values, dtype=torch.float32)
        )

    def transforms_to_dict(self):
        """Serialize the underlying pipelines to a dictionary."""
        return {
            "q": self.transformer_q.to_dict(),
            "y": self.transformer_y.to_dict()
        }

    def invert_transforms_func(self):
        """Return a helper that inverts the fitted transformations."""
        def func(y_arr, q_arr):
            y_arr = self.transformer_y.invert(y_arr)
            q_arr = self.transformer_q.invert(q_arr)
            return y_arr, q_arr
        return func


def _ensure_pipeline(transformer) -> Pipeline:
    """Ensure that ``transformer`` is materialized as a :class:`Pipeline`."""
    if isinstance(transformer, Pipeline):
        return transformer
    try:
        return Pipeline(transformer)
    except Exception as e:
        raise ValueError(f"Invalid {transformer}: {e}")
