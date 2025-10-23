#!/usr/bin/env python3
"""sabi + sanity check -> H5 check 

Validate SAXS or LES HDF5 files used by the AutoFill pipeline.

Checks performed:
- required datasets exist (csv_index, length_nm, concentration(_original), diameter_nm, data_y, data_q / data_wavelength)
- all dataset lengths match csv_index (for 1D) or first axis matches csv_index (for 2D)
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
        required_common = ["csv_index", "length_nm", "diameter_nm", "data_y"]
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

        # If csv_index missing, can't do further length checks
        if check_exists(hdf, "csv_index"):
            csv_len = dataset_len(hdf, "csv_index")
            infos.append((INFO, f"csv_index length: {csv_len}"))

            # Validate lengths for each (1D or 2D first dim)
            for name in list(required_common) + ([conc_found] if conc_found else []) + ([q_found] if q_found else []) + list(extra_cols):
                if name is None:
                    continue
                if not check_exists(hdf, name):
                    continue
                try:
                    dlen = dataset_len(hdf, name)
                except Exception as exc:
                    issues.append((CRITICAL, f"Could not read dataset '{name}': {exc}"))
                    continue
                # For 2D, ensure first dim equals csv_len
                if is_2d(hdf, name):
                    if dlen != csv_len:
                        issues.append((CRITICAL, f"Shape mismatch: dataset '{name}' has first-dim {dlen} but csv_index has {csv_len}"))
                    else:
                        infos.append((INFO, f"Dataset '{name}' matches csv_index length ({csv_len})"))
                else:
                    # 1D
                    if dlen != csv_len:
                        issues.append((CRITICAL, f"Length mismatch: dataset '{name}' length {dlen} vs csv_index {csv_len}"))
                    else:
                        infos.append((INFO, f"Dataset '{name}' matches csv_index length ({csv_len})"))

            # For data_y and q_found, check inner dimension consistency
            if check_exists(hdf, "data_y") and q_found and check_exists(hdf, q_found):
                try:
                    y_shape = hdf["data_y"].shape
                    q_shape = hdf[q_found].shape
                    if len(y_shape) < 2 or len(q_shape) < 2:
                        # it's possible q is 1D per sample? still check first dims
                        infos.append((WARNING, f"data_y or {q_found} do not appear 2D: data_y.shape={y_shape}, {q_found}.shape={q_shape}"))
                    else:
                        if y_shape[1] != q_shape[1]:
                            issues.append((CRITICAL, f"Inner-dimension mismatch between 'data_y' and '{q_found}': {y_shape[1]} vs {q_shape[1]}"))
                        else:
                            infos.append((INFO, f"data_y and {q_found} have consistent inner-dimension: {y_shape[1]}"))
                except Exception as exc:
                    issues.append((WARNING, f"Could not inspect shapes for data_y / {q_found}: {exc}"))

            # NaN and inf checks for critical numeric datasets
            numeric_checks = ["data_y", q_found, "length_nm", "diameter_nm", conc_found]
            for name in numeric_checks:
                if not name or not check_exists(hdf, name):
                    continue
                try:
                    arr = np.array(hdf[name])
                    n_nan, n_inf = count_invalids(arr)
                    if n_nan > 0 or n_inf > 0:
                        issues.append((WARNING, f"Dataset '{name}' contains NaN/inf values: n_nan={n_nan}, n_inf={n_inf}"))
                    else:
                        infos.append((INFO, f"Dataset '{name}' contains no NaN/inf"))
                except Exception as exc:
                    issues.append((WARNING, f"Could not read dataset '{name}' for NaN/inf checks: {exc}"))

        else:
            issues.append((CRITICAL, "Missing 'csv_index' dataset: cannot validate per-sample lengths."))

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

