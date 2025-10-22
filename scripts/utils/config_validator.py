"""Configuration validation utilities for AutoFill training scripts."""

from __future__ import annotations

import argparse
import os
from typing import Any


def validate_config_and_files(config: dict[str, Any], args: argparse.Namespace) -> bool:
    """Validate configuration and check file paths for dry-run mode.

    Args:
        config: The loaded YAML configuration dictionary
        args: Command-line arguments namespace

    Returns:
        bool: True if validation passes, False if errors are found
    """

    print("Validating configuration...")
    print("-" * 50)

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
            print(f"Model type: {model_type} (overridden by --mode argument)")
        else:
            print(f"Model type: {model_type}")

        print(f"Latent dim: {model_config.get('latent_dim', 'default: 128')}")

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
                print(f"HDF5 file found: {h5_file} (overridden by --hdf5_file argument)")
            else:
                print(f"HDF5 file found: {h5_file}")

            # Check file size
            file_size = os.path.getsize(h5_file) / (1024**3)  # GB
            print(f"  File size: {file_size:.2f} GB")

    # Validate training configuration
    if "training" in config:
        training_config = config["training"]

        # Check training indices
        train_indices = training_config.get("array_train_indices")
        val_indices = training_config.get("array_val_indices")

        if not train_indices:
            errors.append("Missing training indices file path")
        elif not os.path.exists(train_indices):
            errors.append(f"Training indices file not found: {train_indices}")
        else:
            print(f"Training indices found: {train_indices}")

        if not val_indices:
            errors.append("Missing validation indices file path")
        elif not os.path.exists(val_indices):
            errors.append(f"Validation indices file not found: {val_indices}")
        else:
            print(f"Validation indices found: {val_indices}")

        # Check training parameters
        num_epochs = training_config.get("num_epochs", 100)
        batch_size = training_config.get("batch_size", 32)
        num_gpus = training_config.get("num_gpus", 1)

        print(f"Training epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Number of GPUs: {num_gpus}")

        # Check GPU availability if requested
        try:
            import torch
            if num_gpus > 0:
                if not torch.cuda.is_available():
                    warnings.append("CUDA not available but num_gpus > 0")
                elif torch.cuda.device_count() < num_gpus:
                    warnings.append(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
                else:
                    print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        except ImportError:
            warnings.append("PyTorch not available - cannot check GPU status")

    # Check conversion dictionary if provided
    conv_dict_path = getattr(args, 'conversion_dict_path', None)
    if conv_dict_path and not os.path.exists(conv_dict_path):
        errors.append(f"Conversion dictionary file not found: {conv_dict_path}")
    elif conv_dict_path:
        print(f"Conversion dictionary found: {conv_dict_path}")

    # Check output directory
    output_dir = config.get("training", {}).get("output_dir", "train_results")
    if not os.path.exists(output_dir):
        warnings.append(f"Output directory does not exist (will be created): {output_dir}")
    else:
        print(f"Output directory exists: {output_dir}")

    print("-" * 50)

    # Report warnings
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()

    # Report errors
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
        print()
        print("Please fix these errors before training.")
        return False
    else:
        print("Configuration validation passed!")
        if getattr(args, 'dry_run', False):
            print("   Ready to start training. Remove --dry-run to begin.")
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
        print(f"Missing {description}")
        return False

    if not os.path.exists(filepath):
        print(f"{description} not found: {filepath}")
        return False

    print(f"{description} found: {filepath}")
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
        if num_gpus > 0:
            if not torch.cuda.is_available():
                warnings.append("CUDA not available but num_gpus > 0")
            elif torch.cuda.device_count() < num_gpus:
                warnings.append(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
            else:
                print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
    except ImportError:
        warnings.append("PyTorch not available - cannot check GPU status")

    return warnings
