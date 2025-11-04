from typing import Any, Dict, Optional, Union
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model types."""
    VAE = "vae"
    PAIR_VAE = "pair_vae"


class ModelSpec(str, Enum):
    """Available model specifications."""
    SAXS = "saxs"
    LES = "les"
    PAIR = "pair"

# --------------------------------------------------------------------------
# Base configs
# --------------------------------------------------------------------------

class BaseModelConfig(BaseModel):
    """Base configuration for a model."""
    type: ModelType = Field(..., description="Model type identifier.")
    spec: ModelSpec  = Field(..., description="Model specification.")
    transforms_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional data transformation config.")


class BaseTrainingConfig(BaseModel):
    """Base configuration for model training."""
    max_lr: float = Field(default=1e-3, gt=0)
    warmup_epochs: int = Field(default=5, ge=0)
    num_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=20, ge=0)
    batch_size: int = Field(default=32, ge=1)
    num_gpus: int = Field(default=1, ge=0)
    num_nodes: int = Field(default=1, ge=0)
    save_every: int = Field(default=1, ge=1)
    output_dir: str = Field(default="train_results")
    plot_train: bool = Field(default=True)
    eta_min: float = Field(default=1e-15, ge=0)
    use_loglog: bool = Field(default=True)
    num_samples: int = Field(default=10, ge=1)
    every_n_epochs: int = Field(default=10, ge=1)
    weighted_loss: bool = Field(default=False)
    weighted_loss_limit_index: Optional[int] = None
    train_indices_path: Optional[Union[str, Path]] = Field(
        default=None, description="Path to file with training data indices.")
    val_indices_path: Optional[Union[str, Path]] = Field(
        default=None, description="Path to file with validation data indices.")
    test_indices_path: Optional[Union[str, Path]] = Field(
        default=None, description="Path to file with test data indices.")

class BaseDatasetConfig(BaseModel):
    """Base configuration for datasets."""
    hdf5_file: Union[str, Path] = Field(..., description="Path to the HDF5 dataset file.")
    conversion_dict: Optional[Union[dict, str, Path]] = Field(
        default=None, description="Conversion mapping or path to conversion dictionary.")
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Filters applied to select specific metadata entries.")
    requested_metadata: Optional[list[str]] = Field(
        default=None, description="List of metadata keys to retrieve from the dataset.")
    use_data_q: bool = Field(default=True, description="Use q-values from the dataset.")


# --------------------------------------------------------------------------
# Single-domain VAE
# --------------------------------------------------------------------------

class VAEModelConfig(BaseModelConfig):
    """Configuration for a single-domain VAE."""
    vae_class: str = Field(..., description="Registered VAE architecture name.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for submodel constructor.")
    beta: float = Field(default=1e-6, ge=0.0, description="KL divergence scaling coefficient.")
    data_q: Optional[Any] = None


class VAETrainingConfig(BaseTrainingConfig):
    """Training configuration for single-domain VAE."""
    sample_frac: float = Field(default=1.0, ge=0.0, le=1.0)


class HDF5DatasetConfig(BaseDatasetConfig):
    """Configuration for single-spectrum HDF5 dataset."""
    transforms_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Transformations applied to q-values."
    )


# --------------------------------------------------------------------------
# Paired-domain (SAXS–LES) VAE
# --------------------------------------------------------------------------

class PairVAEModelConfig(BaseModelConfig):
    """Configuration for paired-domain VAE (SAXS–LES)."""
    spec: ModelSpec = Field(default=ModelSpec.PAIR)
    ckpt_path_saxs: Optional[str] = None
    ckpt_path_les: Optional[str] = None
    lr: float = Field(default=1e-4, ge=0)
    freeze_subvae: bool = Field(default=False)


class PairVAETrainingConfig(BaseTrainingConfig):
    """Training configuration for paired-domain VAE."""
    lambda_param: float = Field(default=0.005, ge=0)
    weight_latent_similarity: float = Field(default=1.0, ge=0)
    weight_saxs2saxs: float = Field(default=1.0, ge=0)
    weight_les2les: float = Field(default=1.0, ge=0)
    weight_saxs2les: float = Field(default=1.0, ge=0)
    weight_les2saxs: float = Field(default=1.0, ge=0)

class PairHDF5DatasetConfig(BaseDatasetConfig):
    """Configuration for paired-spectrum HDF5 dataset."""
