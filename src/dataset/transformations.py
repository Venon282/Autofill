"""Transformation primitives and pipelines used by the datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.logging_utils import get_logger


logger = get_logger(__name__)


class Transformer(ABC):
    """Abstract base class for transformation steps."""

    name: str
    _fitted: bool = False

    @abstractmethod
    def fit(self, data: np.ndarray) -> "Transformer":
        """Fit the transformer parameters and return ``self``."""

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the forward transformation."""

    @abstractmethod
    def invert(self, data: np.ndarray) -> np.ndarray:
        """Invert the transformation."""

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable representation of the transformer configuration."""
        init_params = self.__init__.__code__.co_varnames[1:self.__init__.__code__.co_argcount]
        return {self.name: {k: getattr(self, k) for k in init_params if hasattr(self, k)}}

    def is_fitted(self) -> bool:
        """Return ``True`` when the transformer has been fitted."""
        return self._fitted


class MinMaxScaler(Transformer):
    """Normalize data within the observed minimum and maximum values."""

    name = "MinMaxScaler"

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):

        self.min_val = min_val
        self.max_val = max_val
        self._fitted = not (self.min_val is None or self.max_val is None)

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        if not self._fitted:
            self.min_val = float(np.min(data))
            self.max_val = float(np.max(data))
            self._fitted = True
        if self.min_val >= self.max_val:
            raise ValueError("min_val must be less than max_val.")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform.")
        if self.max_val == self.min_val:
            return np.zeros_like(data)
        return (data - self.min_val) / (self.max_val - self.min_val)

    def invert(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Scaler must be fitted before invert.")
        return data * (self.max_val - self.min_val) + self.min_val

class Padding(Transformer):
    """Pad arrays to a fixed length with constant values."""

    name = "PaddingTransformer"

    def __init__(self, pad_size: int, value: float = 0):
        self.pad_size = pad_size
        self.value = value
        self._fitted = True

    def fit(self, data: np.ndarray) -> "Padding":
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        length = data.shape[-1]
        if length >= self.pad_size:
            return data[..., : self.pad_size]
        pad_width = [(0, 0)] * data.ndim
        pad_width[-1] = (0, self.pad_size - length)
        return np.pad(data, pad_width, constant_values=self.value)

    def invert(self, data: np.ndarray) -> np.ndarray:
        return data

class EnsurePositive(Transformer):
    """Clamp non-positive values to a small epsilon."""

    name = "StrictlyPositiveTransformer"

    def __init__(self, epsilon: float = 1e-15):
        self._fitted = True
        self.epsilon = epsilon

    def fit(self, data: np.ndarray) -> "EnsurePositive":
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.where(data <= 0, self.epsilon, data)

    def invert(self, data: np.ndarray) -> np.ndarray:
        return data

class Log(Transformer):
    """Apply a log transform with epsilon stabilisation."""

    name = "LogTransformer"

    def __init__(self, epsilon: float = 1e-15):
        self._fitted = True
        self.epsilon = epsilon

    def fit(self, data: np.ndarray) -> "Log":
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.log(data + self.epsilon)

    def invert(self, data: np.ndarray) -> np.ndarray:
        return np.exp(data) - self.epsilon


class PreprocessingBase:
    """Base helper that exposes a :class:`Pipeline` through a transformer-like API."""

    pipeline = None

    def __init__(self, pad_size: int, value: float = 0):
        self.pad_size = pad_size
        self.value = value

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Proxy ``transform`` to the internal pipeline."""
        return self.pipeline.transform(data)

    def invert(self, data: np.ndarray) -> np.ndarray:
        """Proxy ``invert`` to the internal pipeline."""
        return self.pipeline.invert(data)

    def fit(self, data: np.ndarray):
        """Fit the underlying pipeline."""
        self.pipeline.fit(data)
        return self

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration of the pipeline."""
        logger.info("Pipeline config: %s", self.pipeline.to_dict())
        return self.pipeline.to_dict()

    def is_fitted(self):
        """Return ``True`` if all steps in the pipeline are fitted."""
        return self.pipeline.is_fitted()


class PreprocessingLES(PreprocessingBase):
    """Preprocessing tailored for LES spectra."""

    name = "PreprocessingLES"

    def __init__(self, pad_size: int, value: float = 0):
        super().__init__(pad_size, value)
        self.pipeline = Pipeline([
            Padding(pad_size, value),
            MinMaxScaler(),
        ])


class PreprocessingSAXS(PreprocessingBase):
    """Preprocessing tailored for SAXS spectra."""

    name = "PreprocessingSAXS"

    def __init__(self, pad_size: int = 54, value: float = 0):
        super().__init__(pad_size, value)
        self.pipeline = Pipeline([
            Padding(pad_size, value),
            EnsurePositive(),
            Log(),
            MinMaxScaler(),
        ])

class PreprocessingQ(PreprocessingBase):
    """Preprocessing for q vectors (padding only)."""

    name = "PreprocessingQ"

    def __init__(self, pad_size: int, value: float = 0):
        super().__init__(pad_size, value)
        self.pipeline = Pipeline([
            Padding(pad_size, value),
        ])


class Pipeline:
    """Composable sequence of :class:`Transformer` steps."""

    transformer_map = {
        MinMaxScaler.name: MinMaxScaler,
        Padding.name: Padding,
        EnsurePositive.name: EnsurePositive,
        Log.name: Log,
        PreprocessingLES.name: PreprocessingLES,
        PreprocessingSAXS.name: PreprocessingSAXS,
        PreprocessingQ.name: PreprocessingQ,
    }

    def __init__(self, config_or_steps: Union[Sequence[Transformer], Dict[str, Dict[str, Any]]]= {}):
        if isinstance(config_or_steps, dict):
            steps: List[Transformer] = []
            for name, params in config_or_steps.items():
                if name not in self.transformer_map:
                    raise ValueError(f"Unknown transformer: {name}")
                steps.append(self.transformer_map[name](**params))
            self.steps = steps
        else:
            self.steps = list(config_or_steps)

    def fit(self, data: Union[np.ndarray, Sequence[np.ndarray]]) -> "Pipeline":
        """Fit each step sequentially using the provided data."""
        array = np.array(data)
        for step in self.steps:
            step.fit(array)
            array = step.transform(array)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply all transformations in order."""
        array = data
        for step in self.steps:
            array = step.transform(array)
        return array

    def batch_transform(self, batch: Sequence[np.ndarray]) -> np.ndarray:
        """Vectorize :meth:`transform` over a batch of inputs."""
        return np.array([self.transform(item) for item in batch])

    def invert(self, data: np.ndarray) -> np.ndarray:
        """Apply the inverse transforms in reverse order."""
        array = data
        for step in reversed(self.steps):
            array = step.invert(array)
        return array

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the pipeline configuration."""
        merged: Dict[str, Any] = {}
        for step in self.steps:
            merged.update(step.get_config())
        return merged

    def is_fitted(self):
        """Return ``True`` if every step reports being fitted."""
        return all([step._fitted for step in self.steps])
