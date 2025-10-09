"""Command-line entry point for training VAE and PairVAE experiments."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.grid_search import GridSearch
from src.model.trainer import TrainPipeline

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
    return parser.parse_args()


def main() -> None:
    """Load configuration overrides and launch training or grid search."""

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.name is not None:
        config["run_name"] = args.name

    if args.material is not None:
        config["dataset"]["metadata_filters"]["material"] = args.material.split(",")
        print(f"Overriding material filter with: {args.material}")

    if args.technique is not None and args.mode != "pair_vae":
        technique_key = args.technique.lower()
        if technique_key not in TRANSFORM_OVERRIDES:
            raise KeyError(f"Unsupported technique override: {args.technique}")
        overrides = TRANSFORM_OVERRIDES[technique_key]
        config["dataset"]["metadata_filters"]["technique"] = args.technique.split(",")
        config["dataset"]["transform"]["y"] = overrides["Y"]
        config["dataset"]["transform"]["q"] = overrides["Q"]
        print(f"Overriding technique filter with: {args.technique}")
        print(f"Applying transform overrides: {overrides}")

    if args.mode == "pair_vae" and args.technique is not None:
        raise ValueError("Technique filters are not supported for pair_vae training mode.")

    config["model"]["type"] = args.mode
    if args.hdf5_file is not None:
        config["dataset"]["hdf5_file"] = args.hdf5_file
    if args.conversion_dict_path is not None:
        config["dataset"]["conversion_dict_path"] = args.conversion_dict_path

    if args.gridsearch:
        gridsearch = GridSearch(config)
        gridsearch.run()
    else:
        trainer = TrainPipeline(config)
        trainer.train()


if __name__ == "__main__":
    main()
