"""
Run inference from a trained checkpoint and export reconstructions (TXT or HDF5).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Add repo root to sys.path for absolute imports when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logging_utils import get_logger
from src.model.configs import ModelType
from src.model.inferencer import run_inference

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for the inference utility."""
    parser = argparse.ArgumentParser(description="Run inference for a trained VAE or PairVAE model.")
    parser.add_argument("-o", "--outputdir", type=str, required=True, help="Directory where outputs will be saved.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the trained checkpoint.")
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Save reconstruction plots alongside data outputs.",
    )
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
    parser.add_argument(
        "-s",
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of the dataset to sample for inference."
             "If <1.0, a random subset will be used."
             "If =1.0, the full dataset will be used."
             "If >1.0, take n=sample_frac samples",
    )
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
    parser.add_argument(
        "--format",
        choices=["txt", "h5"],
        default="txt",
        help="Output format for predictions (default: h5).",
    )
    parser.add_argument(
        "--plot_limit",
        type=int,
        default=10,
        help="Maximum number of plots to save if --plot is enabled.",
    )
    parser.add_argument(
        "--n_jobs_io",
        type=int,
        default=8,
        help="Parallelism for TXT writing.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=1,
        help="Random seed for sampling.",
    )
    parser.add_argument("-p","--no_progressbar",  dest='progressbar', action='store_false')
    return parser.parse_args()


def main() -> None:
    """Dispatch to simplified inference runner."""
    args = parse_args()

    # Load checkpoint metadata to detect model type
    hparams = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_config" in hparams:
        model_type = hparams["model_config"]["type"]
    elif "pairvae_model_config" in hparams:
        model_type = hparams["pairvae_model_config"]["type"]
    else:
        raise ValueError("Checkpoint is missing model configuration.")

    if model_type not in (ModelType.VAE, ModelType.PAIR_VAE):
        raise ValueError(f"Model type {model_type} is not supported for inference.")
    if model_type == ModelType.PAIR_VAE and args.mode is None:
        raise ValueError("Please provide the translation mode for the PairVAE model. (--mode)")
    if model_type == ModelType.VAE and args.mode is not None:
        raise ValueError("The --mode argument is only applicable for PairVAE models.")
    if args.data_dir and args.data_path.endswith(".h5"):
        raise ValueError("Provide either a data path or a data directory, not both.")
    if args.data_dir and not args.data_path.endswith(".csv"):
        raise ValueError("When --data_dir is set, --data_path must point to a CSV file.")

    logger.info("Loading %s model from checkpoint: %s", model_type, args.checkpoint)

    run_inference(
        output_dir=args.outputdir,
        save_plot=args.plot,
        checkpoint_path=args.checkpoint,
        hparams=hparams,
        data_path=args.data_path,
        conversion_dict_path=args.conversion_dict,
        sample_frac=args.sample_frac,
        batch_size=args.batch_size,
        data_dir=args.data_dir or ".",
        output_format=args.format,
        plot_limit=args.plot_limit,
        n_jobs_io=args.n_jobs_io,
        sample_seed=args.sample_seed,
        is_pair=(model_type == "pair_vae"),
        mode=args.mode,
        show_progressbar=args.progressbar,
    )


if __name__ == "__main__":
    main()
