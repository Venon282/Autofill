import argparse
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.inferencer import PairVAEInferencer, VAEInferencer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("--plot", action='store_true', default=False)
    parser.add_argument("-d","--data_path", type=str, required=True,
                        help="Chemin vers le fichier HDF5 ou CSV")
    parser.add_argument("-s", "--sample_frac", type=float, required=False, default=1.0)
    parser.add_argument("-dd","--data_dir", type=str, default=None,
                        help="Répertoire contenant les fichiers de données du csv")
    parser.add_argument('--mode', choices=['les_to_saxs', 'saxs_to_les', 'les_to_les', 'saxs_to_saxs'], required=False, default=None,
                        help='Mode de convertion pour le PairVAE')
    parser.add_argument("-bs", "--batch_size", type=int, default=32,)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = args.checkpoint
    hparams = torch.load(ckpt, map_location='cpu')['hyper_parameters']['config']
    model_type = hparams['model']['type']
    if model_type.lower() not in ['vae', 'pair_vae']:
        raise ValueError(f"Model type {model_type} is not supported for inference.")
    if model_type.lower() == 'pair_vae' and args.mode is None:
        raise ValueError("Please provide the mode for the PairVAE model.")
    if args.data_dir and args.data_path.endswith('.h5'):
        raise ValueError("Please provide either a data path or a data directory, not both.")
    if args.data_dir and not args.data_path.endswith('.csv'):
        raise ValueError("If a data directory is provided, the data path must be a CSV file.")
    print(f"Loading model of type {model_type} from checkpoint: {args.checkpoint}, hparams: {hparams}")
    if model_type == 'vae':
        runner = VAEInferencer(output_dir=args.outputdir, save_plot=args.plot, checkpoint_path=args.checkpoint, data_path=args.data_path, sample_frac=args.sample_frac, hparams=hparams, batch_size=args.batch_size, data_dir=args.data_dir)
    else:
        runner = PairVAEInferencer(output_dir=args.outputdir, save_plot=args.plot, checkpoint_path=args.checkpoint, data_path=args.data_path, sample_frac=args.sample_frac, hparams=hparams, batch_size=args.batch_size, data_dir=args.data_dir)
    runner.infer()
