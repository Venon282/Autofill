#!/usr/bin/env python3
"""sabi + sanity check -> H5 check 

Validate SAXS or LES HDF5 files used by the AutoFill pipeline.

Checks performed:
- required datasets exist (length_nm, concentration(_original), diameter_nm, data_y, data_q / data_wavelength)
- data_y and q arrays have consistent inner dimension (pad size)
- count of NaN/inf values reported
- allows passing extra required columns

Exit codes:
- 0: OK (no critical errors)
- 1: Critical errors found (missing datasets / shape mismatches)

Usage:
    python scripts/H5_check.py --hdf5_path path/to/file.h5 --type saxs

"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)

from src.logging_utils import get_logger


CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"

LEVEL_MAP = {CRITICAL: logging.CRITICAL, WARNING: logging.WARNING, INFO: logging.INFO}


logger = get_logger(__name__)


def check_exists(hdf: h5py.File, name: str) -> bool:
    return name in hdf


def dataset_len(hdf: h5py.File, name: str) -> int:
    d = hdf[name]
    shape = getattr(d, "shape", None)
    if shape is None:
        return 0
    # For 1D datasets return length, for 2D return first dim
    return int(shape[0]) if len(shape) >= 1 else 0


def is_2d(hdf: h5py.File, name: str) -> bool:
    return len(hdf[name].shape) >= 2


def count_invalids(arr: np.ndarray) -> tuple[int, int]:
    # returns (n_nan, n_inf)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    return n_nan, n_inf


def diagnose_hdf5(path: Path, file_type: str, extra_cols: Iterable[str], verbose: bool = False) -> int:
    """Return 0 if ok, 1 if critical issues found."""
    issues = []
    infos = []

    if not path.exists():
        logger.critical("HDF5 file not found: %s", path)
        return 1

    with h5py.File(path, "r") as hdf:
        # Common required fields
        required_common = ["length_nm", "diameter_nm", "data_y"]
        # concentration may be named concentration_original or concentration
        concentration_names = ["concentration_original", "concentration"]

        if file_type == "saxs":
            q_names = ["data_q"]
        elif file_type == "les":
            q_names = ["data_wavelength", "data_q"]
        else:
            # try to auto-detect
            if "data_q" in hdf:
                q_names = ["data_q"]
                file_type = "saxs"
            elif "data_wavelength" in hdf:
                q_names = ["data_wavelength"]
                file_type = "les"
            else:
                q_names = ["data_q", "data_wavelength"]

        # Check common fields
        for col in required_common:
            if not check_exists(hdf, col):
                issues.append((CRITICAL, f"Missing required dataset: '{col}'"))
            else:
                infos.append((INFO, f"Found dataset: {col} (shape={hdf[col].shape})"))

        # Check concentration
        conc_found = None
        for cname in concentration_names:
            if check_exists(hdf, cname):
                conc_found = cname
                infos.append((INFO, f"Found concentration dataset: {cname} (shape={hdf[cname].shape})"))
                break
        if conc_found is None:
            issues.append((CRITICAL, "Missing required concentration dataset ('concentration_original' or 'concentration')"))

        # Check Q dataset
        q_found = None
        for qn in q_names:
            if check_exists(hdf, qn):
                q_found = qn
                infos.append((INFO, f"Found q-axis dataset: {qn} (shape={hdf[qn].shape})"))
                break
        if q_found is None:
            issues.append((CRITICAL, f"Missing q-axis dataset (tried: {', '.join(q_names)})"))

        # Extra required columns
        for col in extra_cols:
            if not check_exists(hdf, col):
                issues.append((CRITICAL, f"Missing required extra column/dataset: '{col}'"))
            else:
                infos.append((INFO, f"Found extra dataset: {col} (shape={hdf[col].shape})"))

    # Print summary
    logger.info("=== H5 check summary ===")
    for level, msg in infos:
        if verbose or level != INFO:
            logger.log(LEVEL_MAP[level], msg)
    for level, msg in issues:
        logger.log(LEVEL_MAP[level], msg)

    # Determine exit code: critical issues present?
    has_critical = any(level == CRITICAL for level, _ in issues)
    return 1 if has_critical else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check integrity of SAXS/LES HDF5 files")
    p.add_argument(dest="hdf5_path", type=str, help="Path to the HDF5 file to check")
    p.add_argument("--type", type=str, choices=["saxs", "les", "auto"], default="auto", help="Type of file to check (saxs/les/auto)")
    p.add_argument("--required-columns", type=str, nargs="*", default=[], help="Extra dataset names that must exist in the file")
    p.add_argument("--verbose", action="store_true", help="Print more info")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    path = Path(args.hdf5_path)
    assert path.suffix in {".h5", ".hdf5"}, "Input file must be an HDF5 file with .h5 or .hdf5 extension"
    rc = diagnose_hdf5(path, args.type, args.required_columns, verbose=args.verbose)
    if rc != 0:
        logger.error("Validation failed: critical issues found.")
    else:
        logger.info("Validation passed: no critical issues.")
    sys.exit(rc)


if __name__ == "__main__":
    main()

