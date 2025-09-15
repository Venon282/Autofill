#!/bin/sh
#SBATCH --job-name=ping_mlflow
#SBATCH --output=log.err
#SBATCH --error=log.err

#SBATCH --partition=RTX6000Node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding

srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/julien/env/autofill_env//bin/python script/03_train.py --gridsearch --config /projects/pnria/julien/autofill/AutoFill/config/pair_vae.yaml --mode pair_vae
