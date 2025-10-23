"""Configuration validation utilities for AutoFill training scripts."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)

from src.logging_utils import get_logger


logger = get_logger(__name__)
DEFAULT_GPUS=0

def validate_config_and_files(config: dict[str, Any], args: argparse.Namespace) -> bool:
    """Validate configuration and check file paths for dry-run mode.

    Args:
        config: The loaded YAML configuration dictionary
        args: Command-line arguments namespace

    Returns:
        bool: True if validation passes, False if errors are found
    """

    logger.info("Validating configuration...")
    logger.info("-" * 50)

    errors = []
    warnings = []

    # Check required top-level parameters
    required_keys = ["experiment_name", "run_name", "model", "dataset", "training"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required parameter: '{key}'")

    # Validate model configuration
    if "model" in config:
        model_config = config["model"]

        # Check if model type is provided either in config or via --mode argument
        model_type = getattr(args, 'mode', None) or model_config.get("type")

        if not model_type:
            errors.append("Missing model type: must be specified either in config 'model.type' or via '--mode' argument")
        elif model_type not in ["vae", "pair_vae"]:
            errors.append(f"Invalid model type: '{model_type}'. Must be 'vae' or 'pair_vae'")

        # Show effective model type (from config or override)
        if getattr(args, 'mode', None):
            logger.info("Model type: %s (overridden by --mode argument)", model_type)
        else:
            logger.info("Model type: %s", model_type)

        logger.info("Latent dim: %s", model_config.get("latent_dim", "default: 128"))

    # Validate dataset configuration
    if "dataset" in config:
        dataset_config = config["dataset"]

        # Check HDF5 file path - can be from config or --hdf5_file argument
        h5_file = getattr(args, 'hdf5_file', None) or dataset_config.get("hdf5_file")
        if not h5_file:
            errors.append("Missing dataset HDF5 file path: must be specified either in config 'dataset.hdf5_file' or via '--hdf5_file' argument")
        elif not os.path.exists(h5_file):
            errors.append(f"HDF5 file not found: {h5_file}")
        else:
            # Show effective HDF5 file path (from config or override)
            if getattr(args, 'hdf5_file', None):
                logger.info(
                    "HDF5 file found: %s (overridden by --hdf5_file argument)", h5_file
                )
            else:
                logger.info("HDF5 file found: %s", h5_file)

            # Check file size
            file_size = os.path.getsize(h5_file) / (1024**3)  # GB
            logger.info("  File size: %.2f GB", file_size)

    # Validate training configuration
    if "training" in config:
        training_keys = ["num_epochs", "batch_size", "num_gpus", "output_dir", 'patience',
                         'output_dir']
        training_config = config["training"]

        # Check training indices
        train_indices = training_config.get("array_train_indices")
        val_indices = training_config.get("array_val_indices")

        if not train_indices:
            errors.append("Missing training indices file path")
        elif not os.path.exists(train_indices):
            errors.append(f"Training indices file not found: {train_indices}")
        else:
            logger.info("Training indices found: %s", train_indices)

        if not val_indices:
            errors.append("Missing validation indices file path")
        elif not os.path.exists(val_indices):
            errors.append(f"Validation indices file not found: {val_indices}")
        else:
            logger.info("Validation indices found: %s", val_indices)

        num_gpus = training_config.get("num_gpus", DEFAULT_GPUS)
        for key in training_keys:
            if key not in training_config:
                raise ValueError(f"Missing required training config key: {key}")
        try:
            import torch
        except ImportError:
            warnings.append("PyTorch not available - cannot check GPU status")
        else:
            if num_gpus > 0:
                if not torch.cuda.is_available():
                    warnings.append("CUDA not available but num_gpus > 0")
                elif torch.cuda.device_count() < num_gpus:
                    warnings.append(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
                else:
                    logger.info("CUDA available with %d GPU(s)", torch.cuda.device_count())

    # Check conversion dictionary if provided
    conv_dict_path = getattr(args, 'conversion_dict_path', None)
    if conv_dict_path and not os.path.exists(conv_dict_path):
        errors.append(f"Conversion dictionary file not found: {conv_dict_path}")
    elif conv_dict_path:
        logger.info("Conversion dictionary found: %s", conv_dict_path)

    # Check output directory
    output_dir = config.get("training", {}).get("output_dir", "train_results")
    if not os.path.exists(output_dir):
        warnings.append(f"Output directory does not exist (will be created): {output_dir}")
    else:
        logger.info("Output directory exists: %s", output_dir)

    logger.info("-" * 50)

    # Report warnings
    if warnings:
        logger.warning("Warnings detected during validation:")
        for warning in warnings:
            logger.warning(" - %s", warning)

    # Report errors
    if errors:
        logger.error(f"Configuration {len(errors)} errors found:")
        for error in errors:
            logger.error(" - %s", error)
        return False
    else:
        logger.info("Configuration validation passed!")
        if getattr(args, 'dry_run', False):
            logger.info("Ready to start training. Remove --dry-run to begin.")
        return True


def validate_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report the result.

    Args:
        filepath: Path to the file to check
        description: Human-readable description of the file

    Returns:
        bool: True if file exists, False otherwise
    """
    if not filepath:
        logger.error("Missing %s", description)
        return False

    if not os.path.exists(filepath):
        logger.error("%s not found: %s", description, filepath)
        return False

    logger.info("%s found: %s", description, filepath)
    return True


def check_gpu_availability(num_gpus: int) -> list[str]:
    """Check GPU availability and return any warnings.

    Args:
        num_gpus: Number of GPUs requested

    Returns:
        list: List of warning messages (empty if no warnings)
    """
    warnings = []

    try:
        import torch
    except ImportError:
        torch = None
        warnings.append("PyTorch not available - cannot check GPU status")
    else:
        if num_gpus > 0:
            if not torch.cuda.is_available():
                warnings.append("CUDA not available but num_gpus > 0")
            elif torch.cuda.device_count() < num_gpus:
                warnings.append(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
            else:
                logger.info("CUDA available with %d GPU(s)", torch.cuda.device_count())

    return warnings
