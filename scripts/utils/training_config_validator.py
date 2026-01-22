"""Configuration validation utilities for training scripts."""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)

from src.logging_utils import get_logger
from src.model.configs import (
    VAEModelConfig,
    PairVAEModelConfig,
    VAETrainingConfig,
    PairVAETrainingConfig,
    HDF5DatasetConfig,
    PairHDF5DatasetConfig,
    ModelType,
)

logger = get_logger(__name__, custom_name="CheckConfig")
DEFAULT_GPUS = 0


def check_config_integrity(config: dict[str, Any], verbose: bool = True) -> bool:
    """Validate configuration consistency and file paths."""
    logger.info("Validating configuration...")
    logger.info("-" * 50)

    errors = []
    warnings = []

    required_sections = ["experiment_name", "model", "training", "dataset"]
    for key in required_sections:
        if key not in config:
            errors.append(f"Missing required section: '{key}'")

    if errors:
        _report(errors=errors)
        return False

    try:
        model_type = ModelType(config["model"]["type"].lower())
    except KeyError:
        errors.append("Missing key 'type' in config['model']")
        _report(errors=errors)
        return False
    except ValueError as e:
        errors.append(f"Invalid model type: {e}")
        _report(errors=errors)
        return False

    try:
        if model_type == ModelType.VAE:
            model_cfg = VAEModelConfig(**{k:v for k, v in config["model"].items() if k!='verbose'}, verbose=verbose)
            train_cfg = VAETrainingConfig(**{k:v for k, v in config["training"].items() if k!='verbose'}, verbose=verbose)
            dataset_cfg = HDF5DatasetConfig(**{k:v for k, v in config["dataset"].items() if k!='verbose'}, verbose=verbose)
        elif model_type == ModelType.PAIR_VAE:
            model_cfg = PairVAEModelConfig(**{k:v for k, v in config["model"].items() if k!='verbose'}, verbose=verbose)
            train_cfg = PairVAETrainingConfig(**{k:v for k, v in config["training"].items() if k!='verbose'}, verbose=verbose)
            dataset_cfg = PairHDF5DatasetConfig(**{k:v for k, v in config["dataset"].items() if k!='verbose'}, verbose=verbose)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        errors.append(f"Invalid configuration structure: {e}")
        _report(errors=errors)
        return False

    h5_file = dataset_cfg.hdf5_file
    if not h5_file or not os.path.exists(h5_file):
        errors.append(f"HDF5 file not found: {h5_file}")
    else:
        size = os.path.getsize(h5_file) / (1024 ** 3)
        logger.info("HDF5 file found: %s (%.2f GB)", h5_file, size)

    if hasattr(dataset_cfg, "conversion_dict") and dataset_cfg.conversion_dict:
        if isinstance(dataset_cfg.conversion_dict, str) and not os.path.exists(dataset_cfg.conversion_dict):
            errors.append(f"Conversion dictionary not found: {dataset_cfg.conversion_dict}")
        else:
            logger.info("Conversion dictionary: %s", dataset_cfg.conversion_dict)

    out_dir = getattr(train_cfg, "output_dir", "train_results")
    if not os.path.exists(out_dir):
        warnings.append(f"Output directory does not exist (will be created): {out_dir}")
    else:
        logger.info("Output directory exists: %s", out_dir)

    train_indices = getattr(train_cfg, "train_indices_path", None)
    val_indices = getattr(train_cfg, "val_indices_path", None)
    test_indices = getattr(train_cfg, "test_indices_path", None)
    for name, path in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        if path:
            if not os.path.exists(path):
                errors.append(f"{name.capitalize()} indices not found: {path}")
            else:
                logger.info("%s indices found: %s", name.capitalize(), path)

    warnings.extend(check_gpu_availability(getattr(train_cfg, "num_gpus", DEFAULT_GPUS)))

    _report(errors=errors, warnings=warnings)
    return len(errors) == 0


def _report(errors=None, warnings=None):
    if warnings:
        logger.warning("Warnings:")
        for w in warnings:
            logger.warning(" - %s", w)
    if errors:
        logger.error("Errors:")
        for e in errors:
            logger.error(" - %s", e)
        logger.error("Configuration validation failed.")
    else:
        logger.info("Configuration validation passed!")


def check_gpu_availability(num_gpus: int) -> list[str]:
    """Check GPU availability and return warnings."""
    warnings = []
    try:
        import torch
    except ImportError:
        warnings.append("PyTorch not available - cannot check GPU status")
        return warnings
    if num_gpus > 0:
        if not torch.cuda.is_available():
            warnings.append("CUDA not available but num_gpus > 0")
        elif torch.cuda.device_count() < num_gpus:
            warnings.append(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
        else:
            logger.info("CUDA available with %d GPU(s)", torch.cuda.device_count())
    return warnings
