"""Utility to merge two compatible HDF5 datasets along the first axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def merge_hdf5(file1: Path, file2: Path, output_file: Path) -> None:
    """Concatenate datasets from ``file1`` and ``file2`` into ``output_file``."""

    with h5py.File(file1, "r") as hdf1, h5py.File(file2, "r") as hdf2, h5py.File(output_file, "w") as hdf_out:
        for key in hdf1.keys():
            data1 = hdf1[key][:]
            data2 = hdf2[key][:]
            merged_data = np.concatenate([data1, data2], axis=0)
            hdf_out.create_dataset(
                key,
                data=merged_data,
                maxshape=(None,) + merged_data.shape[1:],
                dtype=data1.dtype,
            )


def build_parser() -> argparse.ArgumentParser:
    """Expose the merge helper as a small CLI."""

    parser = argparse.ArgumentParser(description="Merge two compatible HDF5 files along the first axis.")
    parser.add_argument("file1", type=str, help="Path to the first HDF5 file.")
    parser.add_argument("file2", type=str, help="Path to the second HDF5 file.")
    parser.add_argument("output", type=str, help="Path to the merged HDF5 file.")
    return parser


def main() -> None:
    """Parse arguments and execute the merge."""

    parser = build_parser()
    args = parser.parse_args()

    merge_hdf5(Path(args.file1), Path(args.file2), Path(args.output))


if __name__ == "__main__":
    main()
