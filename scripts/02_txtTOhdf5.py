"""Convert raw TXT intensity curves and CSV metadata to a single HDF5 file."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class TextToHDF5Converter:
    """Convert a flat CSV catalogue of intensity curves into an HDF5 dataset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_dir: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        pad_size: int = 90,
        hdf_cache: int = 100_000,
        output_hdf5_filename: str = "final_output.h5",
        exclude: Iterable[str] = ("path", "researcher", "date"),
        json_output: str = "conversion_dict.json",
    ) -> None:
        self.dataframe = dataframe
        self.base = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pad_size = pad_size
        self.hdf_cache = hdf_cache
        self.metadata_cols = [col for col in dataframe.columns if col not in set(exclude)]
        self.conversion_dict = {
            col: {} for col in self.metadata_cols if dataframe[col].dtype == object
        }
        final_output = Path(output_hdf5_filename)
        self.final_output_path = (
            final_output
            if final_output.is_absolute()
            else self.output_dir / final_output
        )
        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path = Path(json_output)
        self.json_output_path = (
            json_output_path
            if json_output_path.is_absolute()
            else self.output_dir / json_output_path
        )
        self.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.hdf_data = self._initialize_hdf_data()
        self.hdf_file: h5py.File | None = None

    def _initialize_hdf_data(self) -> list[object]:
        """Allocate the in-memory cache used before writing to disk."""

        return [
            np.zeros((self.hdf_cache, self.pad_size)),
            np.zeros((self.hdf_cache, self.pad_size)),
            np.zeros((self.hdf_cache,)),
            np.zeros((self.hdf_cache,)),
            {col: np.zeros((self.hdf_cache,)) for col in self.metadata_cols},
        ]

    def _create_hdf(self, output_file: Path) -> h5py.File:
        """Create the destination HDF5 file with the required datasets."""

        hdf = h5py.File(output_file, "w")
        hdf.create_dataset("data_q", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("data_y", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("len", (1,), maxshape=(None,))
        hdf.create_dataset("csv_index", (1,), maxshape=(None,))
        for col in self.metadata_cols:
            hdf.create_dataset(col, (1,), maxshape=(None,), dtype=np.float64)
        return hdf

    def _flush_into_hdf5(self, current_index: int, current_size: int) -> None:
        """Persist the cached batch into the HDF5 file."""

        assert self.hdf_file is not None, "HDF5 file must be created before writing."

        self.hdf_file["data_q"].resize((current_index + current_size, self.pad_size))
        self.hdf_file["data_q"][current_index : current_index + current_size, :] = self.hdf_data[0][:current_size, :]
        self.hdf_file["data_y"].resize((current_index + current_size, self.pad_size))
        self.hdf_file["data_y"][current_index : current_index + current_size, :] = self.hdf_data[1][:current_size, :]
        self.hdf_file["len"].resize((current_index + current_size,))
        self.hdf_file["len"][current_index : current_index + current_size] = self.hdf_data[2][:current_size]
        self.hdf_file["csv_index"].resize((current_index + current_size,))
        self.hdf_file["csv_index"][current_index : current_index + current_size] = self.hdf_data[3][:current_size]

        for col in self.metadata_cols:
            self.hdf_file[col].resize((current_index + current_size,))
            self.hdf_file[col][current_index : current_index + current_size] = self.hdf_data[4][col][:current_size]

        self.hdf_file.flush()

    def _load_data_from_file(self, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load a single two-column TXT file containing q and intensity values."""

        if not file_path.exists():
            raise FileNotFoundError(f"Missing data file: {file_path}")

        data_q: list[float] = []
        data_y: list[float] = []
        expected_num_columns = 2

        with open(file_path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "Q", "q", "Q;", "q;")):
                    continue

                if ";" in stripped:
                    tokens = stripped.split(";")
                elif "," in stripped:
                    tokens = stripped.split(",")
                else:
                    tokens = stripped.split()

                try:
                    values = [
                        float(token) if token.lower() != "nan" else float("nan")
                        for token in tokens
                        if token
                    ]
                except ValueError:
                    continue

                if len(values) != expected_num_columns:
                    continue

                clean_values = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in values]
                data_q.append(clean_values[0])
                data_y.append(clean_values[1])

                if len(data_q) > self.pad_size:
                    break

        if not data_q or not data_y:
            raise ValueError(f"No valid rows found in: {file_path}")
        return np.array(data_q, dtype=np.float64), np.array(data_y, dtype=np.float64)

    def _pad_data(self, data_q: np.ndarray, data_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pad or truncate the signals so they match ``pad_size``."""

        if len(data_q) < self.pad_size:
            data_q = np.pad(data_q, (0, self.pad_size - len(data_q)), "constant")
            data_y = np.pad(data_y, (0, self.pad_size - len(data_y)), "constant")
        elif len(data_q) > self.pad_size:
            data_q = data_q[: self.pad_size]
            data_y = data_y[: self.pad_size]
        return data_q, data_y

    def _convert_metadata(self, row: pd.Series) -> dict[str, float]:
        """Map categorical metadata to numeric codes for the HDF5 export."""

        converted: dict[str, float] = {}
        for col in self.metadata_cols:
            value = row[col]

            if pd.isna(value) or value is None:
                converted[col] = -1
                continue

            if isinstance(value, str):
                if value not in self.conversion_dict[col]:
                    self.conversion_dict[col][value] = float(len(self.conversion_dict[col]))
                converted[col] = self.conversion_dict[col][value]
            else:
                converted[col] = float(value)

        return converted

    def convert(self) -> None:
        """Execute the conversion, writing both the HDF5 file and JSON mapping."""

        output_file = self.final_output_path
        self.hdf_file = self._create_hdf(output_file)
        current_size = 0
        current_index = 0

        with tqdm(total=len(self.dataframe), desc="Processing files") as progress:
            for idx, row in self.dataframe.iterrows():
                progress.update(1)
                rel_path = str(row["path"]).lstrip("/")
                file_path = self.base / rel_path
                try:
                    data_q, data_y = self._load_data_from_file(file_path)
                    original_len = len(data_q)
                    data_q, data_y = self._pad_data(data_q, data_y)
                except Exception as error:  # noqa: BLE001 - CLI feedback is important here.
                    data_q = np.zeros((self.pad_size,))
                    data_y = np.zeros((self.pad_size,))
                    original_len = 0
                    warnings.warn(f"Failed to load {row['path']}: {error}")

                metadata = self._convert_metadata(row)
                self.hdf_data[0][current_size] = data_q
                self.hdf_data[1][current_size] = data_y
                self.hdf_data[2][current_size] = original_len
                self.hdf_data[3][current_size] = idx
                for col in self.metadata_cols:
                    self.hdf_data[4][col][current_size] = metadata[col]
                current_size += 1

                if current_size == self.hdf_cache:
                    self._flush_into_hdf5(current_index, current_size)
                    current_index += self.hdf_cache
                    current_size = 0

        if current_size > 0:
            self._flush_into_hdf5(current_index, current_size)

        assert self.hdf_file is not None
        self.hdf_file.close()

        with open(self.json_output_path, "w", encoding="utf-8") as handle:
            json.dump(self.conversion_dict, handle, ensure_ascii=False, indent=2)

        print(f"HDF5 dataset written to: {output_file}")
        print(f"Metadata dictionary written to: {self.json_output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Expose the converter as a small CLI utility used in the tutorials."""

    parser = argparse.ArgumentParser(
        description="Convert CSV metadata and TXT curves into HDF5 format."
    )
    parser.add_argument("--data_csv_path", type=str, required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the raw TXT files.")
    parser.add_argument("--pad_size", type=int, default=500, help="Padding length applied to each curve.")
    parser.add_argument("--output_hdf5_filename", type=str, required=True, help="Destination HDF5 filename.")
    parser.add_argument(
        "--json_output",
        type=str,
        required=True,
        help="Filename for the exported categorical conversion dictionary.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and launch the conversion."""

    parser = build_parser()
    args = parser.parse_args()

    csv_path = Path(args.data_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    converter = TextToHDF5Converter(
        dataframe=dataframe,
        data_dir=args.data_dir,
        output_dir=os.path.dirname(args.output_hdf5_filename) or ".",
        output_hdf5_filename=os.path.basename(args.output_hdf5_filename),
        json_output=args.json_output,
        pad_size=args.pad_size,
    )
    converter.convert()


if __name__ == "__main__":
    main()
