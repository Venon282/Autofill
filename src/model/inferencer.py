# simple_infer.py
from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from joblib import Parallel, delayed

from src.model.vae.pl_vae import PlVAE
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetTXT import TXTDataset
from src.dataset.transformations import Pipeline
from src.logging_utils import get_logger

logger = get_logger(__name__)


# ------------------------- Data structures -------------------------

@dataclass
class ModelOutputs:
    """Container for model outputs in numpy form."""
    y: np.ndarray
    q: np.ndarray
    meta: Dict[str, np.ndarray]


# ------------------------- Utilities (stateless) -------------------------

def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors and collections to the target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def to_numpy(x: Any) -> Any:
    """Convert torch containers to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_numpy(v) for v in x)
    return x


def normalize_batch_arrays(y_batch: np.ndarray, q_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure (y, q) are (B, L). Accepts (B,1,L), (1,L), (L,) and broadcasts q if needed.
    """
    y = np.asarray(y_batch)
    q = np.asarray(q_batch)

    if y.ndim == 3 and y.shape[1] == 1:
        y = np.squeeze(y, axis=1)
    elif y.ndim == 1:
        y = y[None, :]

    if q.ndim == 3 and q.shape[1] == 1:
        q = np.squeeze(q, axis=1)
    elif q.ndim == 1 and y.ndim == 2:
        q = np.broadcast_to(q[None, :], y.shape)
    elif q.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported q shape: {q.shape}")

    if y.ndim != 2 or q.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y={y.shape}, q={q.shape}")
    if y.shape != q.shape:
        raise ValueError(f"Shape mismatch after normalization: y={y.shape}, q={q.shape}")

    return y.astype(np.float64, copy=False), q.astype(np.float64, copy=False)


def filter_plot_arrays(q: np.ndarray, y: np.ndarray, use_loglog: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return masked arrays for plotting; drop non-positive values in log mode."""
    q = np.asarray(q).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if use_loglog:
        mask = (q > 0) & (y > 0) & np.isfinite(q) & np.isfinite(y)
    else:
        mask = np.isfinite(q) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None
    return q[mask], y[mask]


def build_inverse_maps(conversion_dict: Optional[Dict[str, Dict[Any, Any]]]) -> Dict[str, Dict[Any, Any]]:
    """Precompute inverse mapping for metadata conversion."""
    if not conversion_dict:
        return {}
    return {k: {v: kk for kk, v in d.items()} for k, d in conversion_dict.items()}


def convert_metadata(meta_batch: Dict[str, np.ndarray], inv_maps: Dict[str, Dict[Any, Any]]) -> Dict[str, List[Any]]:
    """Apply inverse maps (if any) and listify values."""
    out: Dict[str, List[Any]] = {}
    for k, v in meta_batch.items():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            inv = inv_maps.get(k)
            if inv:
                out[k] = [inv.get(vi, vi) for vi in v.tolist()]
            else:
                out[k] = v.tolist()
        else:
            out[k] = np.array(v).reshape(-1).tolist()
    return out


# ------------------------- Interfaces -------------------------

class ModelAdapter(abc.ABC):
    """Small interface to unify VAE and PairVAE inference."""

    @abc.abstractmethod
    def infer(self, batch: Dict[str, Any]) -> ModelOutputs:
        ...

    @abc.abstractmethod
    def use_loglog(self) -> bool:
        ...

    @abc.abstractmethod
    def signal_name(self) -> str:
        ...


class OutputSink(abc.ABC):
    """Minimal writer interface for TXT/HDF5."""

    @abc.abstractmethod
    def start(self, signal_name: str, use_loglog: bool) -> None:
        ...

    @abc.abstractmethod
    def write_batch(self, batch_idx: int, q: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def finalize(self) -> None:
        ...


class DatasetProvider(abc.ABC):
    """Dataset provider exposing loader and inversion."""

    @abc.abstractmethod
    def loader(self, batch_size: int, indices: np.ndarray) -> DataLoader:
        ...

    @abc.abstractmethod
    def invert(self, y: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abc.abstractmethod
    def inv_maps(self) -> Dict[str, Dict[Any, Any]]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...


# ------------------------- Adapters -------------------------

class VAEAdapter(ModelAdapter):
    """Adapter around PlVAE."""
    def __init__(self, ckpt: str) -> None:
        self.model: PlVAE = PlVAE.load_from_checkpoint(checkpoint_path=ckpt).eval()

    def infer(self, batch: Dict[str, Any]) -> ModelOutputs:
        out = self.model(batch)
        y = to_numpy(out["recon"])
        q = to_numpy(out["data_q"])
        meta = {k: to_numpy(v) for k, v in batch["metadata"].items()}
        return ModelOutputs(y=y, q=q, meta=meta)

    def use_loglog(self) -> bool:
        return bool(getattr(self.model.train_cfg, "use_loglog", False))

    def signal_name(self) -> str:
        spec = self.model.model_cfg.spec
        return "_".join(spec) if isinstance(spec, (list, tuple)) else str(spec)


class PairVAEAdapter(ModelAdapter):
    """Adapter autour de PlPairVAE avec un mode explicite."""
    _methods = {
        "les_to_saxs": "les_to_saxs",
        "saxs_to_les": "saxs_to_les",
        "les_to_les": "les_to_les",
        "saxs_to_saxs": "saxs_to_saxs",
    }

    def __init__(self, ckpt: str, mode: str) -> None:
        if mode not in self._methods:
            raise ValueError(f"Invalid mode '{mode}'.")
        self.model: PlPairVAE = PlPairVAE.load_from_checkpoint(checkpoint_path=ckpt).eval()
        self.mode = mode

    def infer(self, batch: Dict[str, Any]) -> ModelOutputs:
        fn = getattr(self.model, self._methods[self.mode])
        y_pred, q_pred = fn(batch)
        y = to_numpy(y_pred)
        q = to_numpy(q_pred)
        meta = {k: to_numpy(v) for k, v in batch["metadata"].items()}
        return ModelOutputs(y=y, q=q, meta=meta)

    def use_loglog(self) -> bool:
        return self.mode in {"saxs_to_saxs", "les_to_saxs"}

    def signal_name(self) -> str:
        return self.mode


# ------------------------- Providers -------------------------

class H5Provider(DatasetProvider):
    """HDF5 dataset provider with transform/inversion support."""
    def __init__(
        self,
        path: str,
        transformer_q: Pipeline,
        transformer_y: Pipeline,
        conversion_dict: Optional[Dict[str, Dict[Any, Any]]] = None,
        use_data_q: bool = False,
        metadata_filters: Optional[Dict[str, Any]] = None,
        requested_metadata: Optional[List[str]] = None,
    ) -> None:
        self.ds = HDF5Dataset(
            path,
            transformer_q=transformer_q,
            transformer_y=transformer_y,
            conversion_dict=conversion_dict,
            use_data_q=use_data_q,
            metadata_filters=metadata_filters,
            requested_metadata=requested_metadata,
        )
        self._invert = self.ds.invert_transforms_func()
        self._inv_maps = build_inverse_maps(conversion_dict)

    def loader(self, batch_size: int, indices: np.ndarray) -> DataLoader:
        sampler = SubsetRandomSampler(indices)
        return DataLoader(self.ds, batch_size=batch_size, sampler=sampler, pin_memory=True)

    def invert(self, y: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._invert(y, q)

    def inv_maps(self) -> Dict[str, Dict[Any, Any]]:
        return self._inv_maps

    def __len__(self) -> int:
        return len(self.ds)


class CSVProvider(DatasetProvider):
    """CSV-indexed TXT dataset provider."""
    def __init__(self, dataframe_path: str, data_dir: str, transformer_q: Pipeline, transformer_y: Pipeline,
                 conversion_dict: Optional[Dict[str, Dict[Any, Any]]] = None) -> None:
        import pandas as pd
        df = pd.read_csv(dataframe_path).reset_index(drop=True)
        self.ds = TXTDataset(
            dataframe=df,
            data_dir=data_dir,
            transformer_q=transformer_q,
            transformer_y=transformer_y,
        )
        self._invert = self.ds.invert_transforms_func()
        self._inv_maps = build_inverse_maps(conversion_dict)

    def loader(self, batch_size: int, indices: np.ndarray) -> DataLoader:
        sampler = SubsetRandomSampler(indices)
        return DataLoader(self.ds, batch_size=batch_size, sampler=sampler, pin_memory=True)

    def invert(self, y: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._invert(y, q)

    def inv_maps(self) -> Dict[str, Dict[Any, Any]]:
        return self._inv_maps

    def __len__(self) -> int:
        return len(self.ds)


# ------------------------- Sinks -------------------------

class TxtSink(OutputSink):
    """TXT sink with per-sample YAML metadata and optional plotting."""
    def __init__(self, output_dir: Path, plot: bool, plot_limit: int, n_jobs_io: int) -> None:
        self.output_dir = Path(output_dir)
        self.plot = bool(plot)
        self.plot_limit = int(plot_limit)
        self.n_jobs_io = int(n_jobs_io)
        self.signal_name = "recon"
        self.use_loglog_flag = False
        self._n_plotted = 0

    def start(self, signal_name: str, use_loglog: bool) -> None:
        self.signal_name = signal_name
        self.use_loglog_flag = use_loglog
        (self.output_dir / "predictions_txt").mkdir(parents=True, exist_ok=True)
        if self.plot:
            (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    def _maybe_plot(self, q: np.ndarray, y: np.ndarray, title: str, tag: str) -> None:
        if not self.plot or self._n_plotted >= self.plot_limit:
            return
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        qf, yf = filter_plot_arrays(q, y, self.use_loglog_flag)
        if qf is None:
            return
        fig = plt.figure(figsize=(6, 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.set_xscale("log" if self.use_loglog_flag else "linear")
        ax.set_yscale("log" if self.use_loglog_flag else "linear")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("q"); ax.set_ylabel("y"); ax.set_title(title)
        ax.plot(qf, yf, linewidth=1.1)
        fig.savefig(self.output_dir / "plots" / f"{tag}.png")
        plt.close(fig)
        self._n_plotted += 1

    def write_batch(self, batch_idx: int, q: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
        from yaml import safe_dump
        samples_dir = self.output_dir / "predictions_txt"
        b = y.shape[0]
        meta_list = convert_metadata(meta, inv_maps={})
        def _one(i: int) -> None:
            name = f"sample_{batch_idx:06d}_{i:04d}"
            np.savetxt(samples_dir / f"{name}_{self.signal_name}.txt", np.column_stack((q[i], y[i])))
            with (samples_dir / f"{name}_metadata.yaml").open("w", encoding="utf-8") as f:
                safe_dump({k: meta_list[k][i] for k in meta_list}, f)
        Parallel(n_jobs=self.n_jobs_io, prefer="threads")(delayed(_one)(i) for i in range(b))
        for i in range(min(b, max(0, self.plot_limit - self._n_plotted))):
            tag = f"sample_{batch_idx:06d}_{i:04d}_{self.signal_name}"
            self._maybe_plot(q[i], y[i], f"Prediction {self.signal_name}", tag)

    def finalize(self) -> None:
        if self.plot and (self.output_dir / "plots").exists():
            imgs = sorted((self.output_dir / "plots").glob("*.png"))[: self.plot_limit]
            if imgs:
                html = "<h1>Preview</h1>" + "".join(
                    f'<div style="display:inline-block;margin:6px;"><img src="plots/{p.name}" width="320"/></div>'
                    for p in imgs
                )
                (self.output_dir / "preview_report.html").write_text(f"<!doctype html><html><body>{html}</body></html>", encoding="utf-8")


class H5Sink(OutputSink):
    """HDF5 sink with resizable datasets and optional plotting."""
    def __init__(self, output_dir: Path, plot: bool, plot_limit: int) -> None:
        self.output_dir = Path(output_dir)
        self.plot = bool(plot)
        self.plot_limit = int(plot_limit)
        self.signal_name = "recon"
        self.use_loglog_flag = False
        self._n_plotted = 0
        self._h5 = None
        self._q_len: Optional[int] = None
        self._meta_keys: List[str] = []

    def start(self, signal_name: str, use_loglog: bool) -> None:
        import h5py
        self.signal_name = signal_name
        self.use_loglog_flag = use_loglog
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.output_dir / f"predictions_{signal_name}.h5", "w")
        self._h5.attrs["signal"] = signal_name
        if self.plot:
            (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    def _ensure(self, q: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
        import h5py
        if self._q_len is not None:
            return
        y2, q2 = normalize_batch_arrays(y, q)
        self._q_len = int(q2.shape[1])
        self._meta_keys = list(convert_metadata(meta, {}).keys())
        self._h5.create_dataset("q", shape=(0, self._q_len), maxshape=(None, self._q_len), dtype="float64", chunks=True)
        self._h5.create_dataset("y", shape=(0, self._q_len), maxshape=(None, self._q_len), dtype="float64", chunks=True)
        mg = self._h5.create_group("metadata")
        for k in self._meta_keys:
            dt = h5py.string_dtype(encoding="utf-8")
            mg.create_dataset(k, shape=(0,), maxshape=(None,), dtype=dt, chunks=True)

    def write_batch(self, batch_idx: int, q: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
        import numpy as np
        self._ensure(q, y, meta)
        y2, q2 = normalize_batch_arrays(y, q)
        b = y2.shape[0]
        q_ds, y_ds = self._h5["q"], self._h5["y"]
        n = q_ds.shape[0]
        q_ds.resize(n + b, axis=0)
        y_ds.resize(n + b, axis=0)
        q_ds[n:n + b] = q2
        y_ds[n:n + b] = y2

        meta_list = convert_metadata(meta, {})
        for k in self._meta_keys:
            vals = np.asarray(meta_list[k], dtype=object)
            ds = self._h5["metadata"][k]
            ds.resize(n + b, axis=0)
            ds[n:n + b] = vals

        if self.plot and self._n_plotted < self.plot_limit:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            for i in range(min(b, self.plot_limit - self._n_plotted)):
                qf, yf = filter_plot_arrays(q2[i], y2[i], self.use_loglog_flag)
                if qf is None:
                    continue
                fig = plt.figure(figsize=(6, 4), dpi=120)
                ax = fig.add_subplot(111)
                ax.set_xscale("log" if self.use_loglog_flag else "linear")
                ax.set_yscale("log" if self.use_loglog_flag else "linear")
                ax.grid(True, which="both", alpha=0.25)
                ax.set_xlabel("q"); ax.set_ylabel("y"); ax.set_title(f"Prediction {self.signal_name}")
                ax.plot(qf, yf, linewidth=1.1)
                fig.savefig(self.output_dir / "plots" / f"b{batch_idx:06d}_i{i:04d}_{self.signal_name}.png")
                plt.close(fig)
                self._n_plotted += 1

    def finalize(self) -> None:
        if self.plot:
            imgs = sorted((self.output_dir / "plots").glob("*.png"))[: self.plot_limit]
            if imgs:
                html = "<h1>Preview</h1>" + "".join(
                    f'<div style="display:inline-block;margin:6px;"><img src="plots/{p.name}" width="320"/></div>'
                    for p in imgs
                )
                (self.output_dir / "preview_report.html").write_text(f"<!doctype html><html><body>{html}</body></html>", encoding="utf-8")
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None


# ------------------------- Runner -------------------------

class InferenceRunner:
    """
    Minimal orchestrator: adapter + provider + sink. Handles sampling and device.
    """
    def __init__(self, adapter: ModelAdapter, provider: DatasetProvider, sink: OutputSink) -> None:
        self.adapter = adapter
        self.provider = provider
        self.sink = sink

    def run(
        self,
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        sample_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = len(self.provider)
        if sample_frac >= 1.0:
            k = min(int(sample_frac), n)
        else:
            k = max(1, int(round(n * float(sample_frac))))
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=k, replace=False)

        loader = self.provider.loader(batch_size=batch_size, indices=indices)
        self.sink.start(self.adapter.signal_name(), self.adapter.use_loglog())

        logger.info("Dataset size=%d, selected=%d (sample_frac=%.6f)", n, len(indices), float(sample_frac))
        logger.info("Device set to %s", device)
        logger.info("Starting inference...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = move_to_device(batch, device)
                outs = self.adapter.infer(batch)
                y_np, q_np = self.provider.invert(outs.y, outs.q)
                y_np, q_np = normalize_batch_arrays(y_np, q_np)
                self.sink.write_batch(batch_idx, q_np, y_np, outs.meta)

        self.sink.finalize()
        logger.info("Inference results saved.")


# ------------------------- Convenience builder -------------------------

def run_inference(
    *,
    output_dir: str,
    save_plot: bool,
    checkpoint_path: str,
    hparams: Dict[str, Any],
    data_path: str,
    conversion_dict_path: Optional[str] = None,
    sample_frac: float = 1.0,
    batch_size: int = 32,
    data_dir: str = ".",
    output_format: str = "h5",
    plot_limit: int = 100,
    n_jobs_io: int = 4,
    sample_seed: int = 42,
    is_pair: bool = False,
    mode: Optional[str] = None,
) -> None:
    """
    Simple façade keeping the current signature while delegating to the small SOLID core.
    """
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # Conversion dict
    if conversion_dict_path is None:
        conversion_dict = hparams["global_config"].get("conversion_dict")
    else:
        with open(conversion_dict_path, "r", encoding="utf-8") as f:
            conversion_dict = json.load(f)

    # Transforms
    if is_pair:
        tf = hparams["global_config"].get("transforms_data", {})
        transformers = {
            "les": {"q": Pipeline(tf["q_les"]), "y": Pipeline(tf["y_les"])},
            "saxs": {"q": Pipeline(tf["q_saxs"]), "y": Pipeline(tf["y_saxs"])},
        }
        mcfg = {
            "les_to_saxs": {"input": "les", "output": "saxs"},
            "saxs_to_les": {"input": "saxs", "output": "les"},
            "les_to_les": {"input": "les", "output": "les"},
            "saxs_to_saxs": {"input": "saxs", "output": "saxs"},
        }[mode or "les_to_saxs"]
        tq, ty = transformers[mcfg["input"]]["q"], transformers[mcfg["input"]]["y"]
    else:
        # For VAE, re-use model pipelines from config if present; else empty.
        tq = Pipeline(hparams["global_config"].get("transformer_q", []))
        ty = Pipeline(hparams["global_config"].get("transformer_y", []))

    # Provider
    if data_path.endswith((".h5", ".hdf5")):
        provider = H5Provider(
            path=data_path,
            transformer_q=tq, transformer_y=ty,
            conversion_dict=conversion_dict,
            use_data_q=False,
            metadata_filters=hparams["global_config"].get("dataset", {}).get("metadata_filters"),
            requested_metadata=hparams["global_config"].get("dataset", {}).get("requested_metadata"),
        )
    elif data_path.endswith(".csv"):
        provider = CSVProvider(
            dataframe_path=data_path,
            data_dir=data_dir,
            transformer_q=tq, transformer_y=ty,
            conversion_dict=conversion_dict,
        )
    else:
        raise ValueError("Unsupported file format. Use .h5/.hdf5 or .csv")

    # Adapter
    adapter: ModelAdapter = PairVAEAdapter(checkpoint_path, mode or "les_to_saxs") if is_pair else VAEAdapter(checkpoint_path)

    # Sink
    if output_format.lower() == "txt":
        sink: OutputSink = TxtSink(output_dir=output_dir_p, plot=save_plot, plot_limit=plot_limit, n_jobs_io=n_jobs_io)
    elif output_format.lower() == "h5":
        sink = H5Sink(output_dir=output_dir_p, plot=save_plot, plot_limit=plot_limit)
    else:
        raise ValueError("Unsupported output_format. Use 'txt' or 'h5'.")

    # Run
    InferenceRunner(adapter, provider, sink).run(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=batch_size,
        sample_frac=sample_frac,
        seed=sample_seed,
    )
