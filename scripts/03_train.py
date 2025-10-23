"""Command-line entry point for training VAE and PairVAE experiments."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logging_utils import get_logger
from src.model.grid_search import GridSearch
from src.model.trainer import TrainPipeline
from scripts.utils.config_validator import validate_config_and_files


logger = get_logger(__name__)

TRANSFORM_OVERRIDES: dict[str, dict[str, Any]] = {
    "les": {
        "Y": {"PreprocessingLES": {"pad_size": 500}},
        "Q": {"PreprocessingQ": {"pad_size": 500}},
    },
    "saxs": {
        "Y": {"PreprocessingSAXS": {"pad_size": 54}},
        "Q": {"PreprocessingQ": {"pad_size": 54}},
    },
}


def parse_args() -> argparse.Namespace:
    """Return the arguments accepted by the training script."""

    parser = argparse.ArgumentParser(description="Train a VAE or PairVAE model.")
    parser.add_argument("--mode", type=str, choices=["vae", "pair_vae"], help="Model family to train.")
    parser.add_argument("--gridsearch", action="store_true", default=False, help="Run hyper-parameter search instead of a single training run.")
    parser.add_argument("--spec", type=str, choices=["saxs", "les", "pair"], help="Model specification to train.", default=None)
    parser.add_argument("--config", type=str, default="model/VAE/vae_config_saxs.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--name", type=str, default=None, help="Optional run name overriding the config.")
    parser.add_argument("--hdf5_file", type=str, default=None, help="Override the dataset HDF5 file path.")
    parser.add_argument(
        "--conversion_dict_path",
        type=str,
        default=None,
        help="Override the metadata conversion dictionary path.",
    )
    parser.add_argument("--technique", type=str, default=None, help="Filter the dataset to a given acquisition technique.")
    parser.add_argument("--material", type=str, default=None, help="Filter the dataset to a given material label.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Validate configuration and check file paths without starting training.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging output.")
    return parser.parse_args()


def main() -> None:
    """Load configuration overrides and launch training or grid search."""

    args = parse_args()
    if not os.path.exists(args.config):
        logger.error("Configuration file not found: %s", args.config)
        sys.exit(1)
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML configuration: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error loading configuration file: %s", e)
        sys.exit(1)

    if args.name:
        config["run_name"] = args.name
    if args.hdf5_file:
        config.setdefault("dataset", {})["hdf5_file"] = args.hdf5_file
        logger.warning("Overriding HDF5 file path to: %s", args.hdf5_file)
    if args.conversion_dict_path:
        config.setdefault("dataset", {})["conversion_dict_path"] = args.conversion_dict_path
        logger.warning("Overriding conversion dictionary path to: %s", args.conversion_dict_path)
    if args.technique:
        config.setdefault("dataset", {})["technique"] = args.technique
        logger.warning("Filtering dataset to technique: %s", args.technique)
    if args.material:
        config.setdefault("dataset", {})["material"] = args.material
        logger.warning("Filtering dataset to material: %s", args.material)
    if args.spec:
        config["model"]["spec"] = args.spec
        logger.warning("Using model specification from argument: %s", args.spec)


    if not args.mode and "model" in config and "type" in config["model"]:
        args.mode = config["model"]["type"]
        logger.warning("Using model type from config: %s", args.mode)
    config["model"]["type"] = args.mode

    if args.technique and args.technique in TRANSFORM_OVERRIDES:
        config.setdefault("transforms_data", {}).update(TRANSFORM_OVERRIDES[args.technique])

    if not validate_config_and_files(config, args):
        raise SystemExit("Configuration validation failed. Please fix the errors and try again.")

    if not getattr(args, 'dry_run', False):
        try:
            if args.gridsearch:
                logger.info("Starting grid search...")
                grid_search = GridSearch(config)
                grid_search.run()
            else:
                logger.info("Starting training...")
                trainer = TrainPipeline(config, verbose=args.verbose)
                trainer.train()
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.exception("Training failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
