import abc
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetTXT import TXTDataset
from src.dataset.transformations import Pipeline
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch


class BaseInferencer:
    def __init__(self, output_dir, save_plot, checkpoint_path, data_path, sample_frac, hparams, batch_size=32, data_dir="."):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self.load_model(checkpoint_path)
        self.model.to(self.device).eval()
        self.config = hparams
        self.convertion_dict = self.config["convertion_dict"]
        self.data_path = data_path
        self.data_dir = data_dir
        self.sample_frac = sample_frac
        input_dim = self.config["model"]["args"]["input_dim"]
        self.compute_dataset(input_dim)
        self.save_plot = save_plot
        self.use_loglog = self.config["training"]["use_loglog"]

    @abc.abstractmethod
    def compute_dataset(self, input_dim):
        raise NotImplementedError("compute_dataset method should be implemented in subclasses.")

    @abc.abstractmethod
    def load_model(self, path):
        raise NotImplementedError("load_model method should be implemented in subclasses")

    @abc.abstractmethod
    def infer_and_save(self):
        raise NotImplementedError("The infer_and_save method should be implemented in subclasses.")

    def save_pred(self, batch, i, q_arr, y_arrs, metadata):

        # Metadata
        converted_metadata = {}
        for k, v in metadata.items():
            v = v.cpu().numpy().item()
            if self.convertion_dict is not None and k in self.convertion_dict:
                inv_conv = self.convertion_dict[k]
                inv_conv = {v_: k_ for k_, v_ in inv_conv.items()}
                v = inv_conv.get(v, v)                
            converted_metadata[k] = v

        for k, y_arr in y_arrs.items() :
            try:
                if self.format == 'h5':
                    idx = batch['csv_index'][i]
                    name = str(idx.item())
                else:
                    path = batch['path'][i]
                    name = os.path.splitext(os.path.basename(path))[0]
            except KeyError:
                # Handle the case where 'csv_index' or 'path' is not in the batch
                name = f"sample_{i}"
            
            os.makedirs(os.path.join(self.output_dir, f"prediction_{name}"), exist_ok=True)
            txt_filename = os.path.join(self.output_dir, f"prediction_{name}", f"prediction_{k}.txt")
            with open(txt_filename, "w") as txt_file:
                for q_val, recon_val in zip(q_arr, y_arr): 
                    txt_file.write(f"{q_val} {recon_val}\n")

            if self.save_plot : 
                plot_filename = os.path.join(self.output_dir, f"prediction_{name}", f"plot_{k}.png")
                plt.figure()
                if self.use_loglog :
                    plt.loglog(q_arr, y_arr, label='LogLog Prediction')
                else : 
                    plt.plot(q_arr, y_arr, label='Prediction')
                plt.xlabel('q')
                plt.ylabel('y')
                plt.title(f'Prediction {k}')
                plt.grid(True)
                plt.legend()
                plt.savefig(plot_filename)
                plt.close()

        yaml_filename = os.path.join(self.output_dir, f"prediction_{name}", "metadata.yaml")
        with open(yaml_filename, "w") as yaml_file:
            yaml.dump(converted_metadata, yaml_file)

    def infer(self):
        self.infer_and_save()
        print(f"Inference results saved in {self.output_dir}")


class VAEInferencer(BaseInferencer):
    def load_model(self, path):
        return PlVAE.load_from_checkpoint(checkpoint_path=path)

    def infer_and_save(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference per sample"):
                batch = move_to_device(batch, self.device)
                outputs = self.model(batch)
                y_pred = outputs["recon"]
                q_pred = batch['data_q']
                y_pred, q_pred = self.invert(y_pred, q_pred)

                metadata_batch = batch["metadata"]
                for i in range(len(y_pred)):
                    y_arr = y_pred[i].cpu().numpy().flatten()
                    q_arr = q_pred[i].cpu().numpy().flatten()
                    metadata = {k : metadata_batch[k][i] for k in list(metadata_batch.keys())}
                    y_arrs = {"".join(self.config["dataset"]["metadata_filters"].get("technique", "signal")) : y_arr}
                    self.save_pred(batch, i, q_arr, y_arrs, metadata)


    def compute_dataset(self, input_dim):
        if self.data_path.endswith(".h5"):
            self.dataset = HDF5Dataset(
                self.data_path,
                sample_frac=self.sample_frac,
                transformer_q=self.config["transforms_data"]["q"],
                transformer_y=self.config["transforms_data"]["y"],
                metadata_filters=self.config["dataset"]["metadata_filters"],
                conversion_dict=self.config["convertion_dict"],
                requested_metadata=['shape','material','concentration','dimension1','dimension2','opticalPathLength', 'd','h']
            )
            self.format = 'h5'
            self.invert = self.dataset.invert_transforms_func()
        elif self.data_path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(self.data_path)
            df = df[df["technique"].str.lower().isin([t.lower() for t in self.config["dataset"]["metadata_filters"]["technique"]])]
            df = df[df["material"].str.lower().isin([m.lower() for m in self.config["dataset"]["metadata_filters"]["material"]])]
            df = df.reset_index(drop=True)
            self.dataset = TXTDataset(
                dataframe=df,
                data_dir=self.data_dir,
                transformer_q=self.config["transforms_data"]["q"],
                transformer_y=self.config["transforms_data"]["y"],
            )
            self.format = 'csv'
            self.invert = self.dataset.invert_transforms_func()
        else:
            raise ValueError("Unsupported file format. Use .h5 or .csv")


class PairVAEInferencer(BaseInferencer):
    def __init__(self, checkpoint_path, data_path, mode, conversion_dict_path=None, batch_size=32):
        if mode not in {'les_to_saxs', 'saxs_to_les', 'les_to_les', 'saxs_to_saxs'}:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'les_to_saxs' or 'saxs_to_les' or 'les_to_les' or 'saxs_to_saxs'.")
        self.mode = mode
        super().__init__(checkpoint_path, data_path, conversion_dict_path, batch_size)

    def load_model(self, path):
        return PlPairVAE.load_from_checkpoint(path)

    def infer_and_save(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        for batch in tqdm(loader, desc="PairVAE inference"):
            batch = move_to_device(batch, self.device)
            if self.mode == 'les_to_saxs':
                y_pred, q_pred = self.model.les_to_saxs(batch)
            elif self.mode == 'saxs_to_les':
                y_pred, q_pred = self.model.saxs_to_les(batch)
            elif self.mode == 'les_to_les':
                y_pred, q_pred = self.model.les_to_les(batch)
            elif self.mode == 'saxs_to_saxs':
                y_pred, q_pred = self.model.saxs_to_saxs(batch)
            else:
                raise ValueError(f"Unknown inference mode: {self.mode}")

            y_pred, q_pred = self.invert(y_pred, q_pred)

            metadata_batch = batch["metadata"]
            for i in range(len(y_pred)):
                metadata = {k : metadata_batch[k][i] for k in list(metadata_batch.keys())}
                y_arr = y_pred[i].cpu().numpy().flatten()
                q_arr = q_pred[i].cpu().numpy().flatten()
                y_arrs = {self.mode : y_arr}
                self.save_pred(batch, i, q_arr, y_arrs, metadata)

    def compute_dataset(self, input_dim):
        if self.data_path.endswith(".h5"):
            transform_config = self.config.get('transforms_data', {})
            transformer_q_les = Pipeline(transform_config["q_les"])
            transformer_y_les = Pipeline(transform_config["y_les"])
            transformer_q_saxs = Pipeline(transform_config["q_saxs"])
            transformer_y_saxs = Pipeline(transform_config["y_saxs"])

            if self.mode == 'les_to_saxs':
                self.config["dataset"]["metadata_filters"]["technique"] = "les"
                self.dataset = HDF5Dataset(
                    self.data_path,
                    sample_frac=self.sample_frac,
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=self.config['config']["conversion_dict"],
                    transformer_q=transformer_q_les,
                    transformer_y=transformer_y_les,
                    requested_metadata=['shape','material','concentration','dimension1','dimension2','opticalPathLength', 'd','h']
                )
                def invert(y, q):
                    y = transformer_y_saxs.invert(y)
                    q = transformer_q_saxs.invert(q)
                    return y, q
                self.invert = invert

            elif self.mode == 'saxs_to_les':
                self.config["dataset"]["metadata_filters"]["technique"] = "saxs"
                self.dataset = HDF5Dataset(
                    self.data_path,
                    sample_frac=self.sample_frac,
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=self.config['config']["conversion_dict"],
                    transformer_q=transformer_q_saxs,
                    transformer_y=transformer_y_saxs,
                    requested_metadata=['shape','material','concentration','dimension1','dimension2','opticalPathLength', 'd','h']
                )
                def invert(y, q):
                    y = transformer_y_les.invert(y)
                    q = transformer_q_les.invert(q)
                    return y, q
                self.invert = invert

            elif self.mode == 'les_to_les':
                self.config["dataset"]["metadata_filters"]["technique"] = "les"
                self.dataset = HDF5Dataset(
                    self.data_path,
                    sample_frac=self.sample_frac,
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=self.config['config']["conversion_dict"],
                    transformer_q=transformer_q_les,
                    transformer_y=transformer_y_les,
                    requested_metadata=['shape','material','concentration','dimension1','dimension2','opticalPathLength', 'd','h']
                )
                def invert(y, q):
                    y = transformer_y_les.invert(y)
                    q = transformer_q_les.invert(q)
                    return y, q
                self.invert = invert

            elif self.mode == 'saxs_to_saxs':
                self.config["dataset"]["metadata_filters"]["technique"] = "saxs"
                self.dataset = HDF5Dataset(
                    self.data_path,
                    sample_frac=self.sample_frac,
                    metadata_filters=self.config["dataset"]["metadata_filters"],
                    conversion_dict=self.config['config']["conversion_dict"],
                    transformer_q=transformer_q_saxs,
                    transformer_y=transformer_y_saxs,
                    requested_metadata=['shape','material','concentration','dimension1','dimension2','opticalPathLength', 'd','h']
                )
                def invert(y, q):
                    y = transformer_y_saxs.invert(y)
                    q = transformer_q_saxs.invert(q)
                    return y, q
                self.invert = invert

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self.format = 'h5'
            self.invert = self.dataset.inverts_func()

        if self.data_path.endswith(".csv"):
            transform_config = self.config.get('transforms_data', {})
            transformer_q_les = Pipeline(transform_config["q_les"])
            transformer_y_les = Pipeline(transform_config["y_les"])
            transformer_q_saxs = Pipeline(transform_config["q_saxs"])
            transformer_y_saxs = Pipeline(transform_config["y_saxs"])

            if self.mode == 'les_to_saxs':
                self.dataset = TXTDataset(
                    self.data_path,
                    data_dir=self.data_dir,
                    transformer_q=transformer_q_les,
                    transformer_y=transformer_y_les,
                )
                def invert(y, q):
                    y = transformer_y_saxs.invert(y)
                    q = transformer_q_saxs.invert(q)
                    return y, q
                self.invert = invert

            elif self.mode == 'saxs_to_les':
                self.dataset = TXTDataset(
                    self.data_path,
                    data_dir=self.data_dir,
                    transformer_q=transformer_q_saxs,
                    transformer_y=transformer_y_saxs,
                )
                def invert(y, q):
                    y = transformer_y_les.invert(y)
                    q = transformer_q_les.invert(q)
                    return y, q
                self.invert = invert

            elif self.mode == 'saxs_to_saxs':
                self.dataset = TXTDataset(
                    self.data_path,
                    data_dir=self.data_dir,
                    transformer_q=transformer_q_saxs,
                    transformer_y=transformer_y_saxs,
                )
                def invert(y, q):
                    y = transformer_y_saxs.invert(y)
                    q = transformer_q_saxs.invert(q)
                    return y, q
                self.invert = invert

            elif self.mode == 'les_to_les':
                self.dataset = TXTDataset(
                    self.data_path,
                    data_dir=self.data_dir,
                    transformer_q=transformer_q_les,
                    transformer_y=transformer_y_les,
                )
                def invert(y, q):
                    y = transformer_y_les.invert(y)
                    q = transformer_q_les.invert(q)
                    return y, q
                self.invert = invert

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self.format = 'h5'
            self.invert = self.dataset.inverts_func()
        
        else:
            raise ValueError("Unsupported file format. Use .h5")