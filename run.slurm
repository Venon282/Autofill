#!/bin/sh
#SBATCH --job-name=EVAL_AUTOFILL
#SBATCH --output=log_dataset%j.out
#SBATCH --error=log_dataset%j.out
#SBATCH --partition=48CPUNodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/julien/env/autofill_env//bin/python scripts/04_prepare_pairdataset.py \
--saxs_hdf5_path /projects/pnria/caroline/AUTOFILL/data_sept_25/cylinder_saxs_library_no_noise_meta_diameter_metric.h5 \
--les_hdf5_path /projects/pnria/caroline/AUTOFILL/data_sept_25/cylinder_les_meta.h5 \
--dir_output /projects/pnria/caroline/AUTOFILL/data_final3/ \
--output_hdf5_filename pair_data.hdf5 \
--split_test_ratio 0.0