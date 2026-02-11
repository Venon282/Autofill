"""Convert paired LES/SAXS TXT files into a joint HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Dict, Tuple, List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logging_utils import get_logger


logger = get_logger(__name__)


class PairTextToHDF5Converter:
    """Convert aligned LES and SAXS signals into a single HDF5 dataset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_dir: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        pad_size: int = 90,
        hdf_cache: int = 100_000,
        output_hdf5_filename: str = "final_output.h5",
        exclude: Iterable[str] = ("saxs_path", "les_path", "researcher", "date"),
        json_output: str = "conversion_dict.json",
        progressbar = True
    ) -> None:
        self.progressbar = progressbar
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
            final_output if final_output.is_absolute() else self.output_dir / final_output
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
        """Allocate caches for SAXS, LES and metadata batches."""

        return [
            np.zeros((self.hdf_cache, self.pad_size)),
            np.zeros((self.hdf_cache, self.pad_size)),
            np.zeros((self.hdf_cache, self.pad_size)),
            np.zeros((self.hdf_cache, self.pad_size)),
            np.zeros((self.hdf_cache,)),
            np.zeros((self.hdf_cache,)),
            {col: np.zeros((self.hdf_cache,)) for col in self.metadata_cols},
        ]

    def _create_hdf(self, output_file: Path) -> h5py.File:
        """Create the destination HDF5 layout."""

        hdf = h5py.File(output_file, "w")
        hdf.create_dataset("data_q_saxs", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("data_y_saxs", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("data_q_les", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("data_y_les", (1, self.pad_size), maxshape=(None, self.pad_size), dtype=np.float64)
        hdf.create_dataset("len", (1,), maxshape=(None,))
        for col in self.metadata_cols:
            hdf.create_dataset(col, (1,), maxshape=(None,), dtype=np.float64)
        return hdf

    def _flush_into_hdf5(self, current_index: int, current_size: int) -> None:
        """Write the cache content into the HDF5 file."""

        assert self.hdf_file is not None
        self.hdf_file["data_q_saxs"].resize((current_index + current_size, self.pad_size))
        self.hdf_file["data_q_saxs"][current_index : current_index + current_size, :] = self.hdf_data[0][:current_size, :]
        self.hdf_file["data_y_saxs"].resize((current_index + current_size, self.pad_size))
        self.hdf_file["data_y_saxs"][current_index : current_index + current_size, :] = self.hdf_data[1][:current_size, :]
        self.hdf_file["data_q_les"].resize((current_index + current_size, self.pad_size))
        self.hdf_file["data_q_les"][current_index : current_index + current_size, :] = self.hdf_data[2][:current_size, :]
        self.hdf_file["data_y_les"].resize((current_index + current_size, self.pad_size))
        self.hdf_file["data_y_les"][current_index : current_index + current_size, :] = self.hdf_data[3][:current_size, :]
        self.hdf_file["len"].resize((current_index + current_size,))
        self.hdf_file["len"][current_index : current_index + current_size] = self.hdf_data[4][:current_size]
        for col in self.metadata_cols:
            self.hdf_file[col].resize((current_index + current_size,))
            self.hdf_file[col][current_index : current_index + current_size] = self.hdf_data[6][col][:current_size]
        self.hdf_file.flush()

    def _load_data_from_file(self, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load and clean a single LES or SAXS TXT file."""

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

        if not data_q or not data_y:
            raise ValueError(f"No valid rows found in: {file_path}")
        return np.array(data_q, dtype=np.float64), np.array(data_y, dtype=np.float64)

    def _pad_data(self, data_q: np.ndarray, data_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pad or truncate to match the configured ``pad_size``."""

        if len(data_q) < self.pad_size:
            data_q = np.pad(data_q, (0, self.pad_size - len(data_q)), "constant")
            data_y = np.pad(data_y, (0, self.pad_size - len(data_y)), "constant")
        elif len(data_q) > self.pad_size:
            data_q = data_q[: self.pad_size]
            data_y = data_y[: self.pad_size]
        return data_q, data_y

    def _convert_metadata(self, row: pd.Series) -> dict[str, float]:
        """Map categorical metadata values to numeric codes."""

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
        """Execute the conversion and persist both HDF5 and JSON files."""

        output_file = self.final_output_path
        self.hdf_file = self._create_hdf(output_file)
        current_size = 0
        current_index = 0
        if not self.progressbar:
            logger.info("Processing pairs..")
        with tqdm(total=len(self.dataframe), desc="Processing pairs", disable= not self.progressbar) as progress:
            for idx, row in self.dataframe.iterrows():
                progress.update(1)
                file_path_saxs = self.base / str(row["saxs_path"]).lstrip("/")
                file_path_les = self.base / str(row["les_path"]).lstrip("/")

                try:
                    data_q_saxs, data_y_saxs = self._load_data_from_file(file_path_saxs)
                    data_q_les, data_y_les = self._load_data_from_file(file_path_les)
                    original_len = len(data_q_saxs)
                    data_q_saxs, data_y_saxs = self._pad_data(data_q_saxs, data_y_saxs)
                    data_q_les, data_y_les = self._pad_data(data_q_les, data_y_les)
                except Exception as error:  # noqa: BLE001 - CLI feedback is explicit here.
                    warnings.warn(
                        f"Failed to load pair ({row['saxs_path']}, {row['les_path']}): {error}"
                    )
                    continue

                metadata = self._convert_metadata(row)
                self.hdf_data[0][current_size] = data_q_saxs
                self.hdf_data[1][current_size] = data_y_saxs
                self.hdf_data[2][current_size] = data_q_les
                self.hdf_data[3][current_size] = data_y_les
                self.hdf_data[4][current_size] = original_len
                self.hdf_data[5][current_size] = idx
                for col in self.metadata_cols:
                    self.hdf_data[6][col][current_size] = metadata[col]
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

        logger.info("HDF5 dataset written to: %s", output_file)
        logger.info("Metadata dictionary written to: %s", self.json_output_path)


MetadataDict = Dict[int, Tuple[float, float, float]]
ArrayDict = Dict[int, np.ndarray]

class PairingHDF5Converter:
    """
    Convert paired LES and SAXS HDF5 files into a combined HDF5 dataset
    and persist consistent train/validation/test split indices.
    """

    def __init__(
        self,
        saxs_hdf5_path: str | Path,
        les_hdf5_path: str | Path,
        output_dir: str | Path,
        output_filename: str,
        split_val_ratio: float = 0.15,
        split_test_ratio: float = 0.05,
        progressbar=True
    ) -> None:
        """
        Initialize converter with input HDF5 paths and output settings.

        Args:
            saxs_hdf5_path: Path to the SAXS HDF5 file.
            les_hdf5_path: Path to the LES HDF5 file.
            output_dir: Output directory where results will be saved.
            output_filename: Name of the output combined HDF5 file.
            split_val_ratio: Ratio of pairs to include in the validation set.
            split_test_ratio: Ratio of pairs to include in the test set.
        """
        self.progressbar = progressbar
        self.saxs_hdf5_path = Path(saxs_hdf5_path)
        self.les_hdf5_path = Path(les_hdf5_path)
        self.output_dir = Path(output_dir)
        self.output_filename = output_filename
        self.split_val_ratio = float(split_val_ratio)
        self.split_test_ratio = float(split_test_ratio)

        self._validate_split_ratios()

    def _validate_split_ratios(self) -> None:
        """
        Validate the configured validation and test split ratios.
        """
        if self.split_val_ratio < 0.0 or self.split_test_ratio < 0.0:
            raise ValueError("Split ratios must be non-negative.")
        if self.split_val_ratio + self.split_test_ratio >= 1.0:
            raise ValueError("Sum of validation and test split ratios must be less than 1.0.")
        if self.split_val_ratio + self.split_test_ratio >= 0.5:
            logger.warning("High combined split ratio may lead to a small training set.")

    @staticmethod
    def _get_concentration(hdf_obj: h5py.File | h5py.Group) -> np.ndarray:
        """
        Retrieve concentration data from an HDF5 file, supporting multiple key names.

        Returns:
            Concentration array rounded to the nearest thousand and cast to integers.
        """
        if "concentration_original" in hdf_obj:
            concentration = hdf_obj["concentration_original"][:]
        elif "concentration" in hdf_obj:
            concentration = hdf_obj["concentration"][:]
        else:
            raise KeyError(
                "No concentration key found in HDF5 file "
                "(expected 'concentration_original' or 'concentration')."
            )

        return np.round(concentration, -3).astype(int)

    @staticmethod
    def _get_q_values(hdf_obj: h5py.File | h5py.Group) -> np.ndarray:
        """
        Retrieve q/wavelength values from an HDF5 file, supporting multiple key names.
        """
        if "data_q" in hdf_obj:
            return hdf_obj["data_q"][:]
        if "data_wavelength" in hdf_obj:
            return hdf_obj["data_wavelength"][:]
        raise KeyError(
            "No q/wavelength key found in HDF5 file "
            "(expected 'data_q' or 'data_wavelength')."
        )

    def _extract_single_dataset(
        self,
        hdf_obj: h5py.File | h5py.Group,
    ) -> Tuple[MetadataDict, ArrayDict, ArrayDict]:
        """
        Extract metadata and signal dictionaries from a single HDF5 object.

        Returns:
            A tuple of (meta_dict, y_dict, q_dict) with integer keys.
        """
        length_nm = np.asarray(hdf_obj["length_nm"][:], dtype=float)
        diameter_nm = np.asarray(hdf_obj["diameter_nm"][:], dtype=float)
        concentration = self._get_concentration(hdf_obj)
        q_values = self._get_q_values(hdf_obj)
        data_y = np.asarray(hdf_obj["data_y"][:], dtype=float)

        n_samples = length_nm.shape[0]
        indices = range(n_samples)

        meta_dict: MetadataDict = {
            int(idx): (float(length), float(conc), float(diameter))
            for idx, length, conc, diameter in zip(indices, length_nm, concentration, diameter_nm)
        }

        y_dict: ArrayDict = {int(idx): y for idx, y in zip(indices, data_y)}
        q_dict: ArrayDict = {int(idx): q for idx, q in zip(indices, q_values)}

        return meta_dict, y_dict, q_dict

    def _extract_data(
        self,
        hdf_saxs: h5py.File,
        hdf_les: h5py.File,
    ) -> Tuple[
        MetadataDict,
        ArrayDict,
        ArrayDict,
        MetadataDict,
        ArrayDict,
        ArrayDict,
    ]:
        """
        Extract and structure data from SAXS and LES HDF5 files.
        """
        meta_saxs, y_saxs, q_saxs = self._extract_single_dataset(hdf_saxs)
        meta_les, y_les, q_les = self._extract_single_dataset(hdf_les)
        return meta_saxs, y_saxs, q_saxs, meta_les, y_les, q_les

    @staticmethod
    def _build_inverse_mapping(meta_dict: MetadataDict) -> Dict[Tuple[float, float, float], int]:
        """
        Build an inverse mapping from metadata tuples to sample indices.
        """
        return {values: index for index, values in meta_dict.items()}

    def _build_pairs(
        self,
        meta_saxs: MetadataDict,
        meta_les: MetadataDict,
    ) -> List[Tuple[int, int, int]]:
        """
        Match SAXS and LES entries based on identical metadata tuples.

        Returns:
            A list of (pair_index, saxs_idx, les_idx).
        """
        inverse_les = self._build_inverse_mapping(meta_les)
        inverse_saxs = self._build_inverse_mapping(meta_saxs)

        pairs: List[Tuple[int, int, int]] = []
        unpaired_saxs: List[int] = []
        unpaired_les: List[int] = []

        pair_index = 0
        for saxs_idx, saxs_values in meta_saxs.items():
            if saxs_values in inverse_les:
                
                pairs.append((pair_index, saxs_idx, inverse_les[saxs_values]))
                pair_index += 1
            else:
                unpaired_saxs.append(saxs_idx)

        for les_idx, les_values in meta_les.items():
            if les_values not in inverse_saxs:
                unpaired_les.append(les_idx)

        if unpaired_saxs or unpaired_les:
            logger.warning("Some entries could not be paired between SAXS and LES datasets.")

        return pairs

    def _split_pairs(
        self,
        pairs: List[Tuple[int, int, int]],
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """
        Split pairs into train, validation, and test subsets.

        Returns:
            train_pairs, val_pairs, test_pairs as lists of index tuples.
        """
        shuffled_pairs = list(pairs)
        random.shuffle(shuffled_pairs)

        n_total = len(shuffled_pairs)
        n_val = int(self.split_val_ratio * n_total)
        n_test = int(self.split_test_ratio * n_total)
        n_train = n_total - n_val - n_test

        train_pairs = shuffled_pairs[:n_train]
        val_pairs = shuffled_pairs[n_train:n_train + n_val]
        test_pairs = shuffled_pairs[n_train + n_val:]

        logger.info("Training pairs: %d", len(train_pairs))
        logger.info("Validation pairs: %d", len(val_pairs))
        logger.info("Test pairs: %d", len(test_pairs))

        return train_pairs, val_pairs, test_pairs

    def _save_split_indices(
        self,
        train_pairs: List[Tuple[int, int, int]],
        val_pairs: List[Tuple[int, int, int]],
        test_pairs: List[Tuple[int, int, int]],
    ) -> None:
        """
        Persist train/validation/test pair indices to .npy files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            self.output_dir / "train_pairs_saxs_les.npy",
            np.array(train_pairs, dtype=object),
        )
        np.save(
            self.output_dir / "val_pairs_saxs_les.npy",
            np.array(val_pairs, dtype=object),
        )
        if test_pairs:
            np.save(
                self.output_dir / "test_pairs_saxs_les.npy",
                np.array(test_pairs, dtype=object),
            )

    def _compose_hdf5(
        self,
        pairs: List[Tuple[int, int, int]],
        meta_saxs: MetadataDict,
        y_saxs: ArrayDict,
        q_saxs: ArrayDict,
        meta_les: MetadataDict,
        y_les: ArrayDict,
        q_les: ArrayDict,
    ) -> None:
        """
        Combine paired SAXS and LES data into a new HDF5 file.

        Args:
            pairs: List of (pair_index, saxs_idx, les_idx) tuples.
        """
        y_saxs_list: List[np.ndarray] = []
        q_saxs_list: List[np.ndarray] = []
        y_les_list: List[np.ndarray] = []
        q_les_list: List[np.ndarray] = []

        idx_saxs_list: List[int] = []
        idx_les_list: List[int] = []

        length_saxs_list: List[float] = []
        diameter_saxs_list: List[float] = []
        concentration_saxs_list: List[float] = []

        length_les_list: List[float] = []
        diameter_les_list: List[float] = []
        concentration_les_list: List[float] = []
        if not self.progressbar:
            logger.info("Processing pairs..")
        for _, saxs_idx, les_idx in tqdm(pairs, desc="Processing pairs", disable= not self.progressbar):
            y_saxs_list.append(y_saxs[saxs_idx])
            q_saxs_list.append(q_saxs[saxs_idx])
            y_les_list.append(y_les[les_idx])
            q_les_list.append(q_les[les_idx])

            idx_saxs_list.append(saxs_idx)
            idx_les_list.append(les_idx)

            length_saxs, conc_saxs, diameter_saxs = meta_saxs[saxs_idx]
            length_les, conc_les, diameter_les = meta_les[les_idx]

            length_saxs_list.append(length_saxs)
            diameter_saxs_list.append(diameter_saxs)
            concentration_saxs_list.append(conc_saxs)

            length_les_list.append(length_les)
            diameter_les_list.append(diameter_les)
            concentration_les_list.append(conc_les)

        data = {
            "data_y_saxs": np.vstack(y_saxs_list).astype(float),
            "data_q_saxs": np.vstack(q_saxs_list).astype(float),
            "data_y_les": np.vstack(y_les_list).astype(float),
            "data_q_les": np.vstack(q_les_list).astype(float),
            "data_index_saxs": np.asarray(idx_saxs_list, dtype=int),
            "data_index_les": np.asarray(idx_les_list, dtype=int),
            "length_nm_saxs": np.asarray(length_saxs_list, dtype=float),
            "diameter_nm_saxs": np.asarray(diameter_saxs_list, dtype=float),
            "concentration_saxs": np.asarray(concentration_saxs_list, dtype=float),
            "length_nm_les": np.asarray(length_les_list, dtype=float),
            "diameter_nm_les": np.asarray(diameter_les_list, dtype=float),
            "concentration_les": np.asarray(concentration_les_list, dtype=float),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self.output_filename

        with h5py.File(output_path, "w") as f_out:
            for key, val in data.items():
                f_out.create_dataset(key, data=val)

        logger.info("Combined HDF5 saved to: %s", output_path)

    def convert(self) -> None:
        """
        Execute the full conversion pipeline:
        - load HDF5 inputs
        - extract metadata and signals
        - build and split pairs
        - save split indices and combined HDF5 file
        """
        with h5py.File(self.saxs_hdf5_path, "r", swmr=True) as hdf_saxs, h5py.File(
            self.les_hdf5_path, "r"
        ) as hdf_les:
            (
                meta_saxs,
                y_saxs,
                q_saxs,
                meta_les,
                y_les,
                q_les,
            ) = self._extract_data(hdf_saxs, hdf_les)

        pairs = self._build_pairs(meta_saxs, meta_les)
        train_pairs, val_pairs, test_pairs = self._split_pairs(pairs)
        self._save_split_indices(train_pairs, val_pairs, test_pairs)
        self._compose_hdf5(
            pairs=pairs,
            meta_saxs=meta_saxs,
            y_saxs=y_saxs,
            q_saxs=q_saxs,
            meta_les=meta_les,
            y_les=y_les,
            q_les=q_les,
        )


def build_parser() -> argparse.ArgumentParser:
    """Expose the converter via a command-line interface."""

    parser = argparse.ArgumentParser(
        description="Preparation of PAIRVAE dataset in HDF5 format."
    )
    parser.add_argument("--data_csv_path", type=str, help="Path to the paired metadata CSV file.")
    parser.add_argument("--sep", type=str, default=";", help="Field separator used in the CSV file.")
    parser.add_argument("--data_dir", type=str, help="Directory containing the TXT files referenced by the CSV.")
    parser.add_argument("--pad_size", type=int, default=500, help="Padding length applied to each curve.")
    parser.add_argument("--json_output", type=str, default="data.json", help="Destination JSON filename for metadata codes.")

    parser.add_argument("--saxs_hdf5_path", type=str, default="data_saxs.h5", help="HDF5 SAXS path.")
    parser.add_argument("--les_hdf5_path", type=str, default="data_les.h5", help="HDF5 LES path.")
    parser.add_argument("--dir_output", type=str, default="output_pairvae_dataset/", help="Destination for HDF5 and splits.")
    parser.add_argument("--split_val_ratio", type=float, default=0.15, help="Split ratio for validation subset.")
    parser.add_argument("--split_test_ratio", type=float, default=0.05, help="Split ratio for test subset.")

    parser.add_argument("--output_hdf5_filename", type=str, default="pair_data.h5", help="Destination HDF5 filename.")
    parser.add_argument("-p","--no_progressbar",  dest='progressbar', action='store_false')
    return parser



def main() -> None:
    """Parse CLI arguments and launch the paired conversion."""

    parser = build_parser()
    args = parser.parse_args()

    # Determine and log where outputs will be written for user clarity
    if args.data_csv_path is not None :
        out_dir = os.path.dirname(args.output_hdf5_filename) or "."
    else:
        out_dir = args.dir_output
    logger.info("Pair dataset conversion: outputs will be written to '%s'", out_dir)

    if args.data_csv_path is not None :
        csv_path = Path(args.data_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        dataframe = pd.read_csv(csv_path, sep=args.sep)
        converter = PairTextToHDF5Converter(
            dataframe=dataframe,
            data_dir=args.data_dir,
            output_dir=os.path.dirname(args.output_hdf5_filename) or ".",
            output_hdf5_filename=os.path.basename(args.output_hdf5_filename),
            json_output=args.json_output,
            pad_size=args.pad_size,
            progressbar=args.progressbar
        )
    else:
        converter = PairingHDF5Converter(
            saxs_hdf5_path=args.saxs_hdf5_path,
            les_hdf5_path=args.les_hdf5_path,
            output_dir=args.dir_output,
            output_filename=args.output_hdf5_filename,
            split_val_ratio=args.split_val_ratio,
            split_test_ratio=args.split_test_ratio,
            progressbar=args.progressbar
        )
    converter.convert()


if __name__ == "__main__":
    main()
