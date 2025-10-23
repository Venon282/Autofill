"""Small helper to explore metadata distributions inside CSV catalogues."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)

from src.logging_utils import get_logger


logger = get_logger(__name__)


class CSVLoader:
    """Load a CSV file into a list of dictionaries for quick inspection."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)

    def load(self) -> list[dict[str, str]]:
        """Return the CSV content as dictionaries."""

        data: list[dict[str, str]] = []
        with self.filepath.open(mode="r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data


class CSVAnalyzer:
    """Compute dataset statistics for key metadata columns."""

    def __init__(self, data: Iterable[dict[str, str]]) -> None:
        self.data: List[dict[str, str]] = list(data)
        self.total = len(self.data)

    def count_individual(self, column: str) -> dict[str, int]:
        """Count occurrences for a single column."""

        return dict(Counter(row[column] for row in self.data))

    def count_combinations(self) -> dict[tuple[str, str, str], int]:
        """Return the distribution for ``(material, technique, shape)`` triples."""

        combinations = [(row["material"], row["technique"], row["shape"]) for row in self.data]
        return dict(Counter(combinations))

    def count_correlations(self, variable: str) -> int:
        """Count groups where all metadata match except ``variable``."""

        if not self.data:
            return 0

        groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
        headers = [header for header in self.data[0].keys() if header != variable]

        for row in self.data:
            key = tuple(row[header] for header in headers)
            groups[key].add(row[variable])

        count = 0
        for key, values in groups.items():
            if len(values) > 1:
                logger.info(
                    "Correlation for '%s' with metadata key %s: %s", variable, key, values
                )
                count += 1
        return count


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI to explore metadata distributions."""

    parser = argparse.ArgumentParser(description="Analyse metadata distributions in a CSV catalogue.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file to analyse.")
    return parser


def main() -> None:
    """Load the CSV and print summary statistics."""

    parser = build_parser()
    args = parser.parse_args()

    loader = CSVLoader(args.csv_path)
    data = loader.load()

    analyzer = CSVAnalyzer(data)
    if analyzer.total == 0:
        logger.warning("CSV file is empty.")
        return

    count_material = analyzer.count_individual("material")
    count_technique = analyzer.count_individual("technique")
    count_shape = analyzer.count_individual("shape")

    logger.info("== Individual distributions ==")
    logger.info("Material:")
    for key, value in sorted(count_material.items()):
        pct = value / analyzer.total * 100
        logger.info("  - %s: %d (%.2f%%)", key, value, pct)

    logger.info("\nTechnique:")
    for key, value in sorted(count_technique.items()):
        pct = value / analyzer.total * 100
        logger.info("  - %s: %d (%.2f%%)", key, value, pct)

    logger.info("\nShape:")
    for key, value in sorted(count_shape.items()):
        pct = value / analyzer.total * 100
        logger.info("  - %s: %d (%.2f%%)", key, value, pct)

    logger.info("\n== Correlation analysis ==")
    logger.info("Shape correlations: %d", analyzer.count_correlations("shape"))
    logger.info("Material correlations: %d", analyzer.count_correlations("material"))
    logger.info("Technique correlations: %d", analyzer.count_correlations("technique"))


if __name__ == "__main__":
    main()
