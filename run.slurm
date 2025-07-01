#!/bin/sh
#SBATCH --job-name=ping_mlflow
#SBATCH --output=log.err
#SBATCH --error=log.err

#SBATCH --partition=GPUNodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding


srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/caroline/AUTOFILL/env_autofill/bin/python script/05_infer.py \
-c /projects/pnria/julien/autofill/PAIR_VAE/pairvae_grid_20_weight_latent_similarity=0.001_weight_saxs2saxs=0.04_weight_les2les=0.01_weight_saxs2les=0.5_weight_les2saxs=0.5/best.ckpt \
-d /projects/pnria/DATA/AUTOFILL/v2/all_data_v2.h5 \
-cd /projects/pnria/DATA/AUTOFILL/v2/all_data_v2.json \
-o /projects/pnria/caroline/AUTOFILL/test_les2saxs \
--mode les_to_saxs \
-s 0.001 --plot 