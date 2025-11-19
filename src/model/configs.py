from typing import Any, Dict, Optional, Union
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from src.logging_utils import get_logger

logger = get_logger(__name__, custom_name="CONFIG")

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
    transforms_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional data transformation config."
    )
    verbose: bool = Field(default=True, description="If False, disables config warnings.")

    @model_validator(mode="before")
    def warn_extra_fields(cls, values):
        verbose = values.get("verbose", True)
        if not verbose:
            return values

        known_fields = set(cls.model_fields.keys())
        extra = [k for k in values.keys() if k not in known_fields]
        if extra:
            logger.warning(f"[MODEL] Extra fields ignored: {extra}")
        return values

    @model_validator(mode="after")
    def log_default_warnings(self):
        if not getattr(self, "verbose", True):
            return self
        except_fields = ["transforms_data", "data_q", "verbose", "freeze_subvae"]
        provided = getattr(self, "__pydantic_fields_set__", set())
        for name, field in self.model_fields.items():
            if name in except_fields or name in provided:
                continue
            value = getattr(self, name)
            if value == field.default:
                logger.warning(f"[MODEL] Default value used for '{name}': {field.default}")
        return self


class BaseTrainingConfig(BaseModel):
    """Base configuration for model training."""
    max_lr: float = Field(default=1e-4, gt=0)
    warmup_epochs: int = Field(default=5, ge=0)
    num_epochs: int = Field(default=300, ge=1)
    patience: int = Field(default=40, ge=0)
    batch_size: int = Field(default=8, ge=1)
    num_gpus: int = Field(default=1, ge=0)
    num_nodes: int = Field(default=1, ge=0)
    save_every: int = Field(default=1, ge=1)
    output_dir: str = Field(default="train_results")
    plot_train: bool = Field(default=True)
    plot_val: bool = Field(default=True)
    num_workers: int = Field(default=4, ge=0)
    eta_min: float = Field(default=1e-15, ge=0)
    min_delta: float = Field(default=1e-7, ge=0)
    use_loglog: bool = Field(default=True)
    num_samples: int = Field(default=10, ge=1)
    every_n_epochs: int = Field(default=10, ge=1)
    weighted_loss: bool = Field(default=False)
    sample_frac: float = Field(default=1.0, ge=0.0, le=1.0)
    weighted_loss_limit_index: Optional[int] = None
    train_indices_path: Optional[Union[str, Path]] = Field(
        default=None, description="Path to file with training data indices.")
    val_indices_path: Optional[Union[str, Path]] = Field(
        default=None, description="Path to file with validation data indices.")
    test_indices_path: Optional[Union[str, Path]] = Field(
        default=None, description="Path to file with test data indices.")
    validate_config: bool = True
    verbose: bool = Field(default=True, description="If False, disables config warnings.")

    @model_validator(mode="before")
    def warn_extra_fields(cls, values):
        if not values.get("validate_config", True):
            return values
        if not values.get("verbose", True):
            return values

        known_fields = set(cls.model_fields.keys())
        extra = [k for k in values.keys() if k not in known_fields]
        if extra:
            logger.warning(f"[TRAINING] Extra fields ignored: {extra}")
        return values

    @model_validator(mode="after")
    def log_default_warnings(self):
        if not self.validate_config or not self.verbose:
            return self
        except_fields = ["test_indices_path", "verbose", "num_nodes", "plot_train", "plot_val", "save_every", "num_samples", "every_n_epochs", "validate_config"]
        provided = getattr(self, "__pydantic_fields_set__", set())
        for name, field in self.model_fields.items():
            if name in except_fields or name in provided:
                    continue
            value = getattr(self, name)
            if value == field.default:
                if name == "train_indices_path" or name == "val_indices_path":
                    logger.info(f"[TRAINING] No path provided for '{name}'; using full dataset and 80/20 split (train/val).")
                else:
                    logger.warning(f"[TRAINING] Default value used for '{name}': {field.default}")
        return self

class BaseDatasetConfig(BaseModel):
    """Base configuration for datasets."""
    hdf5_file: Union[str, Path] = Field(..., description="Path to the HDF5 dataset file.")
    conversion_dict: Optional[Union[dict, str, Path]] = Field(
        default=None, description="Conversion mapping or path to conversion dictionary.")
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Filters applied to select specific metadata entries.")
    requested_metadata: Optional[list[str]] = Field(
        default=None, description="List of metadata keys to retrieve from the dataset.")
    use_data_q: bool = Field(default=False, description="Use q-values from the dataset.")

    verbose: bool = Field(default=True, description="If False, disables config warnings.")

    @model_validator(mode="before")
    def warn_extra_fields(cls, values):
        verbose = values.get("verbose", True)
        if not verbose:
            return values

        known_fields = set(cls.model_fields.keys())
        extra = [k for k in values.keys() if k not in known_fields]
        if extra:
            logger.warning(f"[DATASET] Extra fields ignored: {extra}")
        return values

    @model_validator(mode="after")
    def log_default_warnings(self):
        if not getattr(self, "verbose", True):
            return self
        except_fields = ["use_data_q", "verbose"]
        provided = getattr(self, "__pydantic_fields_set__", set())
        for name, field in self.model_fields.items():
            if name in except_fields or name in provided:
                    continue
            value = getattr(self, name)
            if value == field.default:
                logger.warning(f"[DATASET] Default value used for '{name}': {field.default}")
        return self


# --------------------------------------------------------------------------
# Single-domain VAE
# --------------------------------------------------------------------------

class VAEModelConfig(BaseModelConfig):
    """Configuration for a single-domain VAE."""
    spec: ModelSpec  = Field(..., description="Model specification.")
    vae_class: str = Field(..., description="Registered VAE architecture name.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for submodel constructor.")
    beta: float = Field(default=1e-7, ge=0.0, description="KL divergence scaling coefficient.")
    data_q: Optional[Any] = None


class VAETrainingConfig(BaseTrainingConfig):
    """Training configuration for single-domain VAE."""


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
    ckpt_path_saxs: Optional[str] = None
    ckpt_path_les: Optional[str] = None
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
