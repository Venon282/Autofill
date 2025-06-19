#!/bin/sh
#SBATCH --job-name=pair_vae
#SBATCH --output=pair_vae.err
#SBATCH --error=pair_vae.err

#SBATCH --partition=RTX6000Node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding


srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/caroline/AUTOFILL/env_autofill/bin/python script/03_train.py --mode pair_vae \
  --config config/pair_vae.yaml --gridsearch