"""Normalize multiple CSV exports into a single dataset.

The original script performed all of its work at import time, which made it
hard to reuse.  This refactor exposes a small processing API with clear
docstrings so the step can be reused from the documentation tutorials and the
command line alike.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm


def extract_dimensions(value: str) -> tuple[float | None, float | None]:
    """Return the diameter and height parsed from a ``dimension`` string."""

    match = re.search(r"d=(\d+(?:\.\d+)?)\s+l=(\d+(?:\.\d+)?)", str(value))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def normalize_path(path_value: str) -> str:
    """Convert Windows-style separators into POSIX paths for portability."""

    normalized = path_value.replace("\\", "/")
    if os.name == "nt":
        return str(Path(normalized))
    return Path(normalized).as_posix()


def process_csv_files(inputs: Iterable[str], output: str, sep: str = ";") -> pd.DataFrame:
    """Load multiple CSV files, clean metadata, and write the merged export."""

    all_rows: list[pd.Series] = []
    for input_file in inputs:
        dataframe = pd.read_csv(input_file, sep=sep, dtype=str)

        if "dimension" in dataframe.columns:
            dataframe["d"], dataframe["h"] = zip(
                *dataframe["dimension"].apply(extract_dimensions)
            )
            dataframe = dataframe.drop(columns=["dimension"])

        if "concentration" in dataframe.columns:
            dataframe["concentration"] = dataframe["concentration"].astype(float)

        for _, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f"Processing {input_file}",
        ):
            row["path"] = normalize_path(row["path"])
            all_rows.append(row)

    merged_df = pd.DataFrame(all_rows)
    merged_df.to_csv(output, index=False, sep=",")
    return merged_df


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the preprocessing step."""

    parser = argparse.ArgumentParser(
        description="Merge raw CSV exports while normalizing metadata."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=str,
        help="Path(s) to the source CSV files.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path where the merged CSV should be written.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=";",
        help="Field separator used in the input CSV files.",
    )
    return parser


def main() -> None:
    """Entry point used by the command line tutorial."""

    parser = build_parser()
    args = parser.parse_args()
    merged_df = process_csv_files(args.inputs, args.output, sep=args.sep)
    print(f"Merged CSV saved to: {args.output}")
    print(f"Total rows written: {len(merged_df)}")


if __name__ == "__main__":
    main()
