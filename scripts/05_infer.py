"""Run inference from a trained checkpoint and export reconstructions."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.inferencer import PairVAEInferencer, VAEInferencer
from src.logging_utils import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for the inference utility."""

    parser = argparse.ArgumentParser(description="Run inference for a trained VAE or PairVAE model.")
    parser.add_argument("-o", "--outputdir", type=str, required=True, help="Directory where outputs will be saved.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--plot", action="store_true", default=False, help="Save reconstruction plots alongside data outputs.")
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        required=True,
        help="Path to the evaluation HDF5 file or metadata CSV.",
    )
    parser.add_argument(
        "-cd",
        "--conversion_dict",
        type=str,
        required=False,
        help="Path to the metadata conversion dictionary if needed.",
    )
    parser.add_argument("-s", "--sample_frac", type=float, default=1.0, help="Fraction of the dataset to sample for inference.")
    parser.add_argument(
        "-dd",
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing raw TXT files referenced by the CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=["les_to_saxs", "saxs_to_les", "les_to_les", "saxs_to_saxs"],
        required=False,
        default=None,
        help="PairVAE translation mode to run.",
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Inference batch size.")
    return parser.parse_args()


def main() -> None:
    """Load checkpoint metadata and dispatch to the appropriate inferencer."""

    args = parse_args()
    checkpoint_path = args.checkpoint

    hparams = torch.load(checkpoint_path, map_location="cpu")["hyper_parameters"]["config"]
    model_type = hparams["model"]["type"]

    if model_type.lower() not in ["vae", "pair_vae"]:
        raise ValueError(f"Model type {model_type} is not supported for inference.")
    if model_type.lower() == "pair_vae" and args.mode is None:
        raise ValueError("Please provide the translation mode for the PairVAE model.")
    if args.data_dir and args.data_path.endswith(".h5"):
        raise ValueError("Provide either a data path or a data directory, not both.")
    if args.data_dir and not args.data_path.endswith(".csv"):
        raise ValueError("When --data_dir is set, --data_path must point to a CSV file.")

    logger.info("Loading %s model from checkpoint: %s", model_type, args.checkpoint)
    if model_type == "vae":
        runner = VAEInferencer(
            output_dir=args.outputdir,
            save_plot=args.plot,
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            conversion_dict_path=args.conversion_dict,
            sample_frac=args.sample_frac,
            hparams=hparams,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )
    else:
        runner = PairVAEInferencer(
            mode=args.mode,
            output_dir=args.outputdir,
            save_plot=args.plot,
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            conversion_dict_path=args.conversion_dict,
            sample_frac=args.sample_frac,
            hparams=hparams,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )
    runner.infer()


if __name__ == "__main__":
    main()
