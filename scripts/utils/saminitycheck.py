"""Quick CLI utility to audit missing files referenced in metadata CSVs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)

from src.logging_utils import get_logger


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the sanity check utility."""

    parser = argparse.ArgumentParser(description="Check that paths in a CSV exist on disk.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the metadata CSV file to verify.")
    parser.add_argument("--basedir", type=str, required=True, help="Base directory prepended to relative paths in the CSV.")
    return parser


def normalize_path(path_value: str) -> Path | None:
    """Return a Path object with consistent separators or ``None`` if empty."""

    normalized = path_value.strip()
    if not normalized or normalized.lower() == "nan":
        return None
    return Path(normalized.replace("\\", "/")).expanduser()


def check_paths(csv_path: Path, base_dir: Path) -> dict[str, list[str]]:
    """Return a mapping of column name to missing relative paths."""

    dataframe = pd.read_csv(csv_path)
    path_columns = [col for col in dataframe.columns if "path" in col.lower() and "optical" not in col.lower()]
    logger.info("Columns containing paths: %s", path_columns)

    missing_per_column: dict[str, list[str]] = {}
    for column in path_columns:
        logger.info("\n--- Checking column: %s ---", column)
        paths = dataframe[column].astype(str).apply(normalize_path)
        missing: list[str] = []

        logger.info("Total files to verify: %d", len(paths))
        preview = [str(p) for p in paths.head(5) if p is not None]
        logger.info("First few entries: %s...", preview)

        for rel_path in tqdm(paths, desc=f"Verifying ({column})"):
            if rel_path is None:
                continue
            cleaned = rel_path.as_posix().lstrip("/")
            full_path = base_dir / cleaned
            if not full_path.exists():
                logger.warning("Missing file: %s", full_path)
                missing.append(cleaned)

        missing_per_column[column] = missing
        logger.info("Missing files for %s: %d / %d", column, len(missing), len(paths))

    return missing_per_column


def main() -> None:
    """Entry point used by the tutorials to validate CSV references."""

    parser = build_parser()
    args = parser.parse_args()

    missing = check_paths(Path(args.csv), Path(args.basedir))

    logger.info("\n--- Summary ---")
    for column, paths in missing.items():
        logger.info("%s: %d missing files", column, len(paths))


if __name__ == "__main__":
    main()
