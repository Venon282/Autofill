Tutorials
=========

This section walks through the typical AutoFill workflow using the command
line utilities found in ``scripts/``. Each tutorial mirrors the README content
but is presented in English for consistency with the rest of the
documentation. The explanations assume you are comfortable with Python but may
be discovering machine-learning tooling such as HDF5 or experiment tracking.
Where relevant we link to :doc:`data_formats` so you can cross-check the
expected outputs.

.. contents::
   :local:
   :depth: 1

CSV Preprocessing
-----------------

Use ``scripts/01_csv_pre_process.py`` to merge and clean the project metadata
CSVs before any conversion or training steps. The command expects one or more
input CSV files separated by semicolons and produces a single normalized CSV
with comma-separated values::

   python scripts/01_csv_pre_process.py \
     data/raw_csv/file1.csv data/raw_csv/file2.csv \
     data/metadata_clean.csv \
     -s ";"

After completion ``data/metadata_clean.csv`` contains the harmonized metadata
that will be consumed by later stages. For large datasets, run the command in a
``tmux`` session so it can continue unattended. Before moving on, validate that
the ``path`` column points to real files using ``scripts/saminitycheck.py``—it
scans every entry and reports the ones that are missing so you do not create
an HDF5 file filled with zeros.

TXT to HDF5 Conversion for VAE
------------------------------

Convert raw ``.txt`` time series into the HDF5 format required by the VAE using
``scripts/02_txtTOhdf5.py``. The script combines the cleaned metadata CSV and
the directory of text files into a single HDF5 file along with a JSON mapping
for categorical metadata::

   python scripts/02_txtTOhdf5.py \
     --data_csv_path data/metadata_clean.csv \
     --data_dir data/txt/ \
     --final_output_file data/all_data.h5 \
     --json_output data/metadata_dict.json \
     --pad_size 900

The resulting ``data/all_data.h5`` file contains the padded sequences, their
lengths, and indices back into the CSV, while ``data/metadata_dict.json`` maps
categorical labels to their encoded values. See :doc:`data_formats` for a full
description of the layout and a short Python snippet that prints sample values.

Training the VAE
----------------

Kick off VAE training with ``scripts/03_train.py`` by pointing it to the HDF5
file produced in the previous step and the desired YAML configuration::

   python scripts/03_train.py \
     --config configs/vae.yml \
     --data data/all_data.h5 \
     --metadata_json data/metadata_dict.json

Adjust the configuration file to control training hyperparameters such as
learning rate, batch size, and model architecture. Beginners should keep an eye
on ``input_dim`` and the padding size: they must match the value used during the
conversion step, otherwise the dataloaders will raise a shape mismatch error.

TXT to HDF5 Conversion for PairVAE
----------------------------------

When working with paired modalities, prepare the dataset with
``scripts/04_pair_txtTOhdf5.py``. The options mirror the single-modality
conversion but generate the inputs expected by PairVAE::

   python scripts/04_pair_txtTOhdf5.py \
      --data_csv_path data/metadata_clean.csv \
      --data_dir data/txt/ \
      --final_output_file data/pair_all_data.h5 \
      --json_output data/pair_metadata_dict.json \
      --pad_size 900

Training the PairVAE
--------------------

Train the PairVAE model with ``scripts/03_train.py`` by switching the mode to
``pair_vae``. The remaining arguments follow the same pattern as the VAE
trainer::

   python scripts/03_train.py \
     --mode pair_vae \
     --config configs/pairvae.yml \
     --hdf5_file data/pair_all_data.h5 \
     --conversion_dict_path data/pair_metadata_dict.json

The PairVAE configuration file controls the dual-branch architecture,
reconstruction weights, and latent alignment behavior. Just like with the VAE,
ensure that the padding size in the configuration matches the value used during
conversion for both modalities.

Running Inference
-----------------

After training, evaluate the saved checkpoints with ``scripts/05_infer.py``. The
script loads a trained model, runs inference on the specified dataset, and
stores outputs in the chosen directory::

   python scripts/05_infer.py \
     --outputdir outputs/inference \
     --checkpoint runs/pairvae/checkpoints/last.ckpt \
     --data_path data/pair_all_data.h5

Validation Metrics and SASFit Analysis
--------------------------------------

Once you have a trained checkpoint and a prepared HDF5 file, consolidate the
results with ``scripts/06_val_metrics.py``. The utility computes reconstruction
metrics such as MAE, MSE, RMSE, and R² on a random subset of the dataset and can
run SASFit-based fittings to recover physical parameters (diameter, length, and
concentration)::

   python scripts/06_val_metrics.py \
      --checkpoint logs/vae_model.ckpt \
      --data_path data/all_data.h5 \
      --conversion_dict data/metadata_dict.json \
      --outputdir outputs/vae_metrics \
      --eval_percentage 0.1 \
      --sasfit_percentage 0.005

Key options to keep in mind:

* ``--mode`` is required for PairVAE checkpoints and must be either
  ``les_to_saxs`` or ``saxs_to_saxs``; the remaining modes do not expose ground
  truth reconstructions.
* ``--eval_percentage`` controls the share of samples used for reconstruction
  metrics (``0.1`` = 10%). ``--sasfit_percentage`` works the same way for the
  SASFit pass and should generally stay low to keep runtime manageable.
* ``--n_processes`` lets you cap the amount of CPU parallelism leveraged during
  SASFit fitting. By default, the script uses all available cores minus one.

The command writes a comprehensive report to the output directory:

* ``validation_metrics.yaml`` contains every metric along with the hyperparameter
  snapshot loaded from the checkpoint.
* ``metrics_summary.txt`` is a text digest that you can paste into lab notes or
  emails.
* ``reconstruction_metrics_detailed.csv`` lists sample-level scores when
  reconstructions were evaluated.

Grid Search (Advanced)
----------------------

For hyperparameter tuning, launch the integrated grid search utility::

   python scripts/03_train.py \
     --mode vae \
     --gridsearch \
     --config configs/vae.yml

The search configuration defines the parameter ranges, evaluation metrics, and
resource limits for the sweep. Monitor the generated logs to inspect the
results and identify the best-performing setup. When a run looks promising,
commit the YAML configuration and the metadata JSON so you can replay the exact
conditions later.
