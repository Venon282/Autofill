"""Convert paired LES/SAXS TXT files into a joint HDF5 dataset."""

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
import random


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
        hdf.create_dataset("csv_index", (1,), maxshape=(None,))
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
        self.hdf_file["csv_index"].resize((current_index + current_size,))
        self.hdf_file["csv_index"][current_index : current_index + current_size] = self.hdf_data[5][:current_size]
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

        with tqdm(total=len(self.dataframe), desc="Processing pairs") as progress:
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

        print(f"HDF5 dataset written to: {output_file}")
        print(f"Metadata dictionary written to: {self.json_output_path}")

class PairingHDF5Converter:
    """Convert HDF5 LES and SAXS files into a single HDF5 dataset with splits."""

    def __init__(self, saxs_hdf5_path, les_hdf5_path, dir_output, output_hdf5_filename, split_train_ratio=0.8):

        self.hdf_saxs = h5py.File(saxs_hdf5_path, 'r', swmr=True)
        self.hdf_les = h5py.File(les_hdf5_path, 'r')

        self.dir_output = dir_output
        self.output_hdf5 = output_hdf5_filename

        self.split_train_ratio = split_train_ratio

    def _extract_data(self):
        csv_index_saxs = self.hdf_saxs["csv_index"][:]
        length_nm_saxs = self.hdf_saxs["length_nm"][:]
        concentration_original_saxs = self.hdf_saxs["concentration_original"][:]
        concentration_original_saxs = np.round(concentration_original_saxs, -3).astype(int)
        diameter_nm_saxs = self.hdf_saxs["diameter_nm"][:]

        dict_saxs = {int(csv_index): (float(length), float(conc), float(radius))
            for csv_index, length, conc, radius in zip(
                csv_index_saxs, length_nm_saxs, concentration_original_saxs, diameter_nm_saxs)}
        dict_saxs_y_values = {int(csv_index): y_values
            for csv_index, y_values in zip(
                csv_index_saxs, self.hdf_saxs["data_y"][:])}
        dict_saxs_q_values = {
            int(csv_index): q_values
            for csv_index, q_values in zip(
                csv_index_saxs, self.hdf_saxs["data_q"][:])}

        csv_index_les = self.hdf_les["csv_index"][:]
        length_nm_les = self.hdf_les["length_nm"][:]
        concentration_original_les = self.hdf_les["concentration"][:]
        concentration_original_les = np.round(concentration_original_les, -3).astype(int)
        diameter_nm_les = self.hdf_les["diameter_nm"][:]

        dict_les = {int(csv_index): (float(length), float(conc), float(radius))
            for csv_index, length, conc, radius in zip(
                csv_index_les, length_nm_les, concentration_original_les, diameter_nm_les)}
        dict_les_y_values = {int(csv_index): y_values
            for csv_index, y_values in zip(
                csv_index_les, self.hdf_les["data_y"][:])}
        dict_les_q_values = {
            int(csv_index): q_values
            for csv_index, q_values in zip(
                csv_index_les, self.hdf_les["data_wavelength"][:])}

        return dict_saxs, dict_saxs_y_values, dict_saxs_q_values, dict_les, dict_les_y_values, dict_les_q_values

    def _split_dataset(self, dict_saxs, dict_les):

        inverse_dict_les = {v:k for k,v in dict_les.items()}
        inverse_dict_saxs = {v:k for k,v in dict_saxs.items()}

        # Liste des paires
        paires_saxs_les = []
        sans_paire_saxs = []
        sans_paire_les = []
        index_pair = 0
        for k_saxs, v_saxs in dict_saxs.items():
            if v_saxs in inverse_dict_les:
                paires_saxs_les.append((index_pair, k_saxs, inverse_dict_les[v_saxs]))
                index_pair+=1
            else:
                sans_paire_saxs.append(k_saxs)
        for k_les, v_les in dict_les.items():
            if v_les not in inverse_dict_saxs:
                sans_paire_les.append(k_les)

        if len(sans_paire_les)>0 or len(sans_paire_saxs)>0:
            print("ERROR PAIRING : Some data were not paired")

        # Splitting
        random.shuffle(paires_saxs_les)
        # Calcul de la taille du train
        n_train = int(self.split_train_ratio * len(paires_saxs_les))
        # Split
        train = paires_saxs_les[:n_train]
        val = paires_saxs_les[n_train:]

        print("Nombre de paires train :", len(train))
        print("Nombre de paires val   :", len(val))
        # Exemple : train et val sont des listes de paires

        os.makedirs(self.dir_output, exist_ok=True)
        np.save(f"{self.dir_output}/train_(pair_saxs_les).npy", np.array(train, dtype=object))
        np.save(f"{self.dir_output}/val_(pair_saxs_les).npy", np.array(val, dtype=object))

        return paires_saxs_les

    def _compose_hdf5(self, paires_saxs_les, dict_saxs_y_values, dict_saxs_q_values, dict_les_y_values, dict_les_q_values):

        combined_data_y_saxs = []
        combined_data_q_saxs = []
        combined_data_y_les = []
        combined_data_q_les = []
        csv_index_pair_list = []
        csv_index_saxs_list = []
        csv_index_les_list = []

        # Parcourir les triplets avec barre de progression
        for index_pair, csv_idx_saxs, csv_idx_les in tqdm(paires_saxs_les, desc="Traitement des paires"):
            # Ajouter les données aux listes
            combined_data_y_saxs.append(dict_saxs_y_values[csv_idx_saxs])
            combined_data_q_saxs.append(dict_saxs_q_values[csv_idx_saxs])
            combined_data_y_les.append(dict_les_y_values[csv_idx_les])
            combined_data_q_les.append(dict_les_q_values[csv_idx_les])
            csv_index_pair_list.append(index_pair)
            csv_index_saxs_list.append(csv_idx_saxs)
            csv_index_les_list.append(csv_idx_les)

        # Convertir en arrays
        combined_data_y_saxs = np.vstack(combined_data_y_saxs).astype(float)
        combined_data_q_saxs = np.vstack(combined_data_q_saxs).astype(float)
        combined_data_y_les = np.vstack(combined_data_y_les).astype(float)
        combined_data_q_les = np.vstack(combined_data_q_les).astype(float)
        csv_index_pair_list = np.array(csv_index_pair_list)
        csv_index_saxs_list = np.array(csv_index_saxs_list)
        csv_index_les_list = np.array(csv_index_les_list)

        # Créer le nouveau HDF5
        with h5py.File(f"{self.dir_output}/{self.output_hdf5}", 'w') as f_out:
            f_out.create_dataset('data_y_saxs', data=combined_data_y_saxs)
            f_out.create_dataset('data_q_saxs', data=combined_data_q_saxs)
            f_out.create_dataset('data_y_les', data=combined_data_y_les)
            f_out.create_dataset('data_q_les', data=combined_data_q_les)
            f_out.create_dataset('csv_index', data=csv_index_pair_list)
            f_out.create_dataset('csv_index_saxs', data=csv_index_saxs_list)
            f_out.create_dataset('csv_index_les', data=csv_index_les_list)

    def convert(self):
        """Execute the concatenation and save both HDF5 file and npy split files."""

        dict_saxs, dict_saxs_y_values, dict_saxs_q_values, dict_les, dict_les_y_values, dict_les_q_values = self._extract_data()
        paires_saxs_les = self._split_dataset(dict_saxs, dict_les)
        self._compose_hdf5(paires_saxs_les, dict_saxs_y_values, dict_saxs_q_values, dict_les_y_values, dict_les_q_values)


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
    parser.add_argument("--dir_output", type=str, default="output_pairvar_dataset/", help="Destination for HDF5 and splits.")
    parser.add_argument("--split_train_ratio", type=float, default=0.8, help="Split ratio for training subset.")

    parser.add_argument("--output_hdf5_filename", type=str, default="data.h5", help="Destination HDF5 filename.")
    return parser


def main() -> None:
    """Parse CLI arguments and launch the paired conversion."""

    parser = build_parser()
    args = parser.parse_args()

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
        )
    else:
        converter = PairingHDF5Converter(
            saxs_hdf5_path=args.saxs_hdf5_path,
            les_hdf5_path=args.les_hdf5_path,
            dir_output=args.dir_output,
            output_hdf5_filename=args.output_hdf5_filename,
            split_train_ratio=args.split_train_ratio,
        )
    converter.convert()


if __name__ == "__main__":
    main()
