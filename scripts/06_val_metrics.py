#!/usr/bin/env python3
"""Compute validation metrics for VAE and PairVAE checkpoints."""

import argparse
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.validation import ValidationEngine
from src.validation.utils.utils import display_validation_results
from pprint import pprint

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Validation metrics calculator")
    parser.add_argument("-c", "--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("-d", "--data_path", required=True, help="HDF5 file path")
    parser.add_argument("-o", "--outputdir", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["les_to_saxs", "saxs_to_saxs", "les_to_les", "saxs_to_les"], help="PairVAE mode")
    parser.add_argument("--signal_length", type=int, default=1000, help="Forced signal length")
    parser.add_argument("-cd", "--conversion_dict", help="Metadata conversion file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument("--eval_percentage", type=float, default=0.05, help="Dataset fraction for reconstruction metrics")
    parser.add_argument("--fit_percentage", type=float, default=0.05, help="Dataset fraction for fit metrics")

    parser.add_argument("--qmin_fit", type=float, default=0.001, help="Q min fitting")
    parser.add_argument("--qmax_fit", type=float, default=0.5, help="Q max fitting")
    parser.add_argument("--factor_scale_to_conc", type=float, default=20878, help="Scale to concentration factor")
    parser.add_argument("--n_processes", type=int, help="Number of processes for fit metrics")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Entry point for validation script."""

    load_dotenv()
    args = parse_args()

    engine = ValidationEngine(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.outputdir,
        conversion_dict_path=args.conversion_dict,
        batch_size=args.batch_size,
        eval_percentage=args.eval_percentage,
        fit_percentage=args.fit_percentage,
        qmin_fit=args.qmin_fit,
        qmax_fit=args.qmax_fit,
        factor_scale_to_conc=args.factor_scale_to_conc,
        n_processes=args.n_processes,
        random_state=args.random_state,
        signal_length=args.signal_length,
        mode=args.mode,
    )

    results = engine.run()
    print("Validation completed.")
    display_validation_results(results)


if __name__ == "__main__":
    main()
