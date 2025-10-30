Tutorials
=========

The tutorials walk through the complete AutoFill workflow with beginner-friendly
instructions. Every step refers to a script under ``scripts/`` and explains:

* the purpose of the command,
* the arguments you need to provide, and
* the files or console output you should expect.

Follow the numbered order below if you are running the pipeline for the first
time.

.. important::
    If you already have HDF5 files for both SAXS and LES, you can skip Step 0
    (TXT -> HDF5) and the TXT-to-HDF5 conversion steps described in Step 3.
    If you need paired train/validation splits that are reused by both VAE and
    PairVAE training, run Step 1 (Pairing HDF5 Converter) to create the
    ``.npy`` split files and reference them in your YAML configs.

.. contents::
   :local:
   :depth: 1

.. important::
    * Use a virtual environment with all dependencies installed (see
      :doc:`getting_started`).
    * You can replace ``python`` with ``uv run`` if you installed the
      project with ``uv``. This runs the scripts within the virtual environment
      without activating it explicitly.

.. _step1-preprocess:

Step 0 – (Optional) Convert all TXT files to HDF5
----------------------------------------------------------------------

This optional step shows the minimal sequence to convert raw TXT files into
single-modality HDF5 files for both SAXS and LES. Run these sub-steps when
you start from raw text curves; if you already have HDF5 files you can skip
this entire Step 0.

0.1 Preprocess CSV metadata (script `01_csv_pre_process.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Why:** Normalise and merge raw CSV exports into a single clean metadata
file that points to your TXT files.

.. code-block:: bash

   python scripts/01_csv_pre_process.py \
     data/raw_csv/file1.csv data/raw_csv/file2.csv \
     data/metadata_clean.csv \
     --sep ";"

0.2 Convert SAXS TXT files to HDF5 (script `02_txtTOhdf5.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Why this matters:** the neural networks expect a fixed-size HDF5 dataset
rather than loose text files.

Please study step 3 before training VAE in order to prepare the data for PairVAE.

**Command**

.. code-block:: bash

   python scripts/02_txtTOhdf5.py \
     --data_csv_path data/metadata_clean.csv \
     --data_dir data/txt/ \
     --output_hdf5_filename data/all_data.h5 \
     --json_output data/metadata_dict.json \
     --pad_size 900

**Arguments**

* ``--data_csv_path`` – path to the cleaned CSV from Step 1.
* ``--data_dir`` – folder containing the raw ``.txt`` curves referenced in the
  CSV ``path`` column.
* ``--output_hdf5_filename`` – destination for the generated HDF5 file. The parent
  folders will be created automatically.
* ``--json_output`` – filename that will store the categorical metadata mapping
  (used later by the models to decode labels).
* ``--pad_size`` – optional. Sets the number of points kept per curve. Shorter
  curves are padded with zeros; longer curves are truncated.

**Outputs**

* ``data/all_data.h5`` – holds the padded ``q`` and ``intensity`` arrays, the
  original lengths, and the CSV row indices.
* ``data/metadata_dict.json`` – JSON dictionary mapping categorical values to
  numeric codes.
* Console messages showing the destination paths of the HDF5 and JSON files.

.. tip::

   * If the script prints an error about a missing TXT file, double-check that the
     ``path`` column in your CSV uses relative paths starting from ``data_dir``.
   * A pad size that is too small will crop information; too large may slow down
     training. Start with 900 for SAXS data.

0.3 Convert LES TXT files to HDF5 (script `02_txtTOhdf5.py`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Why this matters:** PairVAE requires paired inputs (e.g. LES and SAXS) stored
in a single HDF5 file.

PairTextToHDF5Converter
***********************

If you have a pairing file that contains paths to the new TXT SAXS and TXT LES data, use **PairTextToHDF5Converter**.
This option should be used when you have new SAXS and LES datasets to avoid data leakage — meaning the individual VAEs have not seen this data before.
The pairing file should be a panda dataframe with columns `saxs_path` and `les_path`.

**Command**

.. code-block:: bash

   python scripts/04_prepare_pairdataset.py \
     --data_csv_path data/metadata_clean.csv \
     --data_dir data/txt/ \
     --output_hdf5_filename data/pair_all_data.h5 \
     --json_output data/pair_metadata_dict.json \
     --pad_size 900

**Arguments**

* ``--data_csv_path`` – CSV containing metadata for both modalities. It can be
  the same file as Step 1 if paired paths are present.
* ``--sep`` – optional delimiter override if the CSV is not comma-separated.
* ``--data_dir`` – directory with the text files referenced by the CSV.
* ``--output_hdf5_filename`` – target HDF5 file holding both modalities.
* ``--json_output`` – metadata mapping dedicated to the paired dataset.
* ``--pad_size`` – shared padding applied to both modalities.

**Outputs**

* ``data/pair_all_data.h5`` – HDF5 dataset containing paired arrays and helper
  metadata.
* ``data/pair_metadata_dict.json`` – categorical mapping aligned with the paired
  dataset.
* Console logs indicating where the files were written and how many pairs were
  processed.

Step 1 – Pairing HDF5 Converter (create paired dataset + shared splits)
-----------------------------------------------------------------------

If you want to use the same datasets for training both the VAE and the PairVAE, use PairingHDF5Converter.
This converter creates data splits (training/validation) before training the VAEs. These same splits are then reused for training the PairVAE, ensuring that training and validation subsets never overlap.
The splits are save as ``.npy`` files that you need to inform in the ``.yaml`` training files as `array_train_indices` and `array_val_indices` for BOTH VAE training and PairVAE training to ensure that the rights splits are always used.
There 
**Command**

.. code-block:: bash

   python scripts/04_prepare_pairdataset.py \
     --saxs_hdf5_path DATA/cylinder_saxs_library_no_noise_meta_diameter_metric.h5 \
     --les_hdf5_path DATA/cylinder_les_meta.h5 \
     --dir_output DATA \
     --output_hdf5_filename pair_all_data.hdf5

**Arguments**

* ``--saxs_hdf5_path`` – HDF5 SAXS path.
* ``--les_hdf5_path`` – HDF5 LES path.
* ``--dir_output`` – directory with the npy and HDF5 files are stored.
* ``--output_hdf5_filename`` – name of HDF5 file.
* ``--split_val_ratio`` – split ratio >0 and <1.. => default 0.15.
* ``--split_test_ratio`` – split ratio >0 and <1.. => default 0.05.

**Outputs**

* ``data/pair_all_data.h5`` – HDF5 dataset containing paired arrays and helper
  metadata (written to the directory specified by ``--dir_output``).
* ``train_pairs_saxs_les.npy`` – array containing triplets (pair_idx, saxs_idx, les_idx). Saved in ``--dir_output``. Note: saved with dtype=object; load with ``np.load(..., allow_pickle=True)``.
* ``val_pairs_saxs_les.npy`` – array containing triplets (pair_idx, saxs_idx, les_idx). Saved in ``--dir_output``. Note: saved with dtype=object; load with ``np.load(..., allow_pickle=True)``.

**Required HDF5 keys created/expected**

The combined HDF5 file created by the converter (and required by `PairHDF5Dataset`) must contain the following datasets:

* ``data_q_saxs``
* ``data_y_saxs``
* ``data_q_les``
* ``data_y_les``

.. tip::
   * When combining modalities, keep a consistent folder structure so both files in
     a pair can be found relative to ``--data_dir``.
   * Use a pad size that accommodates the longest modality to avoid truncation.

Using the splits in your YAML configuration
-------------------------------------------

The converter's ``.npy`` split files must be referenced in your training
YAMLs so both the standalone VAE runs and the PairVAE run use the exact same
subsets. In your configuration files, set the dataset fields that reference
precomputed index arrays. For example:

.. code-block:: yaml

   dataset:
     h5_file_path: "data/pair_all_data.h5"
     array_train_indices: "data/train_pairs_saxs_les.npy"
     array_val_indices: "data/val_pairs_saxs_les.npy"

.. important::
    Supply the same ``array_train_indices`` and ``array_val_indices``
    paths for BOTH the VAE training configuration and the PairVAE configuration.
    This ensures the VAE models are trained and validated on the identical sample
    subsets that the PairVAE later uses to align modalities.
    ``array_train_indices`` and ``array_val_indices`` contains h5 indices.

Outputs
~~~~~~~

* ``data/pair_all_data.h5`` – HDF5 dataset containing paired arrays and
  helper metadata.
* ``data/pair_metadata_dict.json`` – categorical mapping aligned with the
  paired dataset.
* ``train_indices.npy`` and ``val_indices.npy`` – index arrays you must
  reference in your YAML files as ``array_train_indices`` and
  ``array_val_indices`` for both VAE and PairVAE training.
* Console logs indicating where the files were written and how many pairs
  were processed.

.. tip::
   * Large pairing runs may take a few minutes; run them in a detached
     session (tmux/screen) when working on remote machines.


Step 2 – Train the VAE
----------------------

**Why this matters:** trains a Variational Autoencoder that can reconstruct a
single modality (for example SAXS).

.. admonition:: Check the HDF5 file and launch a simple VAE
   :class: tip

   Before starting a full training, you can quickly verify that your HDF5 file
   is correct using the utility script ``scripts/utils/H5_check.py``. Example:

   .. code-block:: bash

      # checks the structure and a few entries in the HDF5
      python scripts/utils/H5_check.py DATA/cylinder_les_meta.h5

   After verification, launch a simple VAE training (``vae`` mode).
   To test the configuration without starting the actual training, use
   ``--dry-run``; to start real training, remove ``--dry-run``:

   .. code-block:: bash

      # validate the configuration and paths without training
      python scripts/03_train.py --config config/vae_les.yaml --mode vae --hdf5_file DATA/cylinder_les_meta.h5 --dry-run

      # start a simple VAE training
      python scripts/03_train.py --config config/vae_les.yaml --mode vae --hdf5_file DATA/cylinder_les_meta.h5

   See :doc:`configuration` for complete parameter reference and examples.

**Command**

.. code-block:: bash

   python scripts/03_train.py --config config/vae_saxs.yaml

**Command-line arguments**

The training script accepts several command-line arguments that can override configuration file settings:

* ``--config`` – **Required**: Path to YAML configuration file containing all training parameters
* ``--mode`` – **Optional**: Model family to train (``"vae"`` or ``"pair_vae"``). Overrides the ``model.type`` setting in the configuration file.
* ``--hdf5_file`` – **Optional**: Override the dataset HDF5 file path. Overrides the ``dataset.h5_file_path`` setting in the configuration file.
* ``--name`` – **Optional**: Override the run name from the configuration file.
* ``--conversion_dict_path`` – **Optional**: Override the metadata conversion dictionary path.
* ``--technique`` – **Optional**: Filter the dataset to a specific acquisition technique.
* ``--material`` – **Optional**: Filter the dataset to a specific material label.
* ``--gridsearch`` – **Optional**: Run hyperparameter search instead of single training.
* ``--dry-run`` – **Optional**: Validate configuration and check file paths without starting training.
* ``--verbose`` – **Optional**: Enable detailed logging output for debugging.

**Examples**

Basic training with configuration file:

.. code-block:: bash

   python scripts/03_train.py --config config/vae_saxs.yaml

Override the HDF5 file path:

.. code-block:: bash

   python scripts/03_train.py --config config/vae_saxs.yaml --hdf5_file data/custom_data.h5

Override the model type:

.. code-block:: bash

   python scripts/03_train.py --config config/base_config.yaml --mode pair_vae

Validate configuration before training:

.. code-block:: bash

   python scripts/03_train.py --config config/vae_saxs.yaml --dry-run

**Important note about overrides**

The ``--hdf5_file`` and ``--mode`` arguments allow you to override configuration settings without modifying the YAML file. This is particularly useful for:

- Testing different datasets with the same model configuration
- Switching between VAE and PairVAE models with similar parameters
- Running experiments with different data files in batch scripts


**Example configuration structure**::

   experiment_name: "saxs_vae_experiment"
   run_name: "baseline_run"

   model:
     type: "vae"  # Can be overridden with --mode
     latent_dim: 128
     # ... more parameters

   dataset:
     h5_file_path: "data/all_data.h5"  # Can be overridden with --hdf5_file
     # ... more parameters

   training:
     num_epochs: 100
     batch_size: 32
     # ... more parameters


**Example configuration structure**::

   experiment_name: "saxs_vae_experiment"
   run_name: "baseline_run"

   model:
     type: "vae"  # Can be overridden with --mode
     latent_dim: 128
     # ... more parameters

   dataset:
     h5_file_path: "data/all_data.h5"  # Can be overridden with --hdf5_file
     # ... more parameters

   training:
     num_epochs: 100
     batch_size: 32
     # ... more parameters

For complete configuration examples and all available parameters, refer to :doc:`configuration`.

**Outputs**

* Training logs and checkpoints in the directory specified by your configuration
* TensorBoard logs (when no MLFlow URI is provided) or MLFlow tracking
* Best model checkpoint saved as ``best.ckpt``
* Configuration backup saved as ``config_model.yaml``

**Training outputs**

.. note::
   After training, AutoFill generates comprehensive outputs including model checkpoints, configuration files, and visualization plots for monitoring and analysis.

**Directory structure**

After training, you'll find the following structure:

.. code-block:: none

   {output_dir}/{experiment_name}/{run_name}/
   ├── best.ckpt                    # Best model checkpoint
   ├── config_model.yaml           # Complete configuration used
   ├── train_indices.npy           # Training data indices
   ├── val_indices.npy             # Validation data indices
   └── inference_results/           # Generated plots
       ├── val_plot.png            # Validation plots
       └── train_plot.png          # Training plots (if enabled)

{output_dir}/{experiment_name}/
    └── tensorboard_logs/
        └── {run_name}
            ├── hparams.yaml
            └── events.out.tfevents.XXXXX


**Model checkpoints**

``best.ckpt``
    Contains the model state with the best validation loss, including:

    - Model weights and biases
    - Optimizer state
    - Learning rate scheduler state
    - Training epoch information
    - Validation metrics
    - Transformation pipeline details
    - Data Q range used during training

``lightning_logs/version_*/checkpoints/``
    Contains periodic checkpoints saved during training (controlled by ``save_every``).

**Configuration files**

``config_model.yaml``
    Complete configuration file used for the training run, including:

    - All hyperparameters (including defaults)
    - Dataset transformation pipeline
    - Model architecture details
    - Training settings
    - Metadata conversion dictionaries

**Visualization and monitoring**

**TensorBoard** (when no MLFlow URI is provided):

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir=train_results

   # View in browser at http://localhost:6006

**MLFlow** (when MLFlow URI is provided):

.. code-block:: bash

   # MLFlow UI (if running local server)
   mlflow ui --host 0.0.0.0 --port 5000

   # View in browser at http://localhost:5000

**Available metrics:**

- ``train_loss``, ``val_loss``: Training and validation losses
- ``val_mae_recon``: Mean absolute error for reconstruction
- ``sasfit_*``: SAS fitting metrics (if enabled)
- Learning rate schedules
- Model hyperparameters

**Inference plots**

Generated plots include:

- **Reconstruction comparisons**: Original vs. reconstructed signals
- **Cross-modal translations**: (PairVAE only) SAXS↔LES conversions
- **Validation samples**: Regular monitoring during training
- **Training samples**: (if ``plot_train: true``) Training set examples

.. tip::

   * **Start with the configuration guide**: :doc:`configuration` contains complete examples
   * Monitor training with: ``tensorboard --logdir=train_results``
   * Check your HDF5 file structure matches your configuration parameters


Step 3 – Train the PairVAE
--------------------------

**Why this matters:** trains the cross-domain model that aligns LES and SAXS
representations.

.. important::
   Like VAE training, PairVAE requires a proper configuration file.
   See :doc:`configuration` for PairVAE-specific parameters and complete examples.

**Command**

.. code-block:: bash

   python scripts/03_train.py --config config/pair_vae.yaml

**Configuration requirements**

PairVAE training requires a YAML configuration file that specifies:

* **Model parameters**: ``type: "pair_vae"``, encoder/decoder architectures for both modalities
* **Dataset settings**: paired HDF5 file paths, modality-specific transformations
* **Training options**: cross-modal loss weights, batch size, epochs
* **Logging setup**: output directories, visualization settings

**Example PairVAE configuration structure**::

   experiment_name: "multimodal_saxs_les"
   run_name: "pair_vae_baseline"

   model:
     type: "pair_vae"
     latent_dim: 128
     encoder_saxs_layers: [512, 256, 128]
     encoder_les_layers: [256, 128, 64]
     # ... more parameters

   dataset:
     h5_file_path: "data/pair_all_data.h5"
     # ... modality-specific parameters

   training:
     num_epochs: 200
     batch_size: 32
     # ... more parameters

For complete PairVAE configuration examples and all available parameters, refer to :doc:`configuration`.

**Arguments**

* ``--config`` – **Required**: Path to YAML configuration file with PairVAE-specific parameters

**Outputs**

* Training logs and checkpoints in the directory specified by your configuration
* Cross-modal reconstruction plots and metrics
* Best model checkpoint with both modality encoders/decoders
* Configuration backup for reproducibility

.. note::
   For detailed information about training outputs, directory structure, and monitoring options, see the training outputs section in Step 3.

Step 4 – Run inference
----------------------

**Why this matters:** generates reconstructions or translations from a trained
checkpoint and saves the results for inspection.

**Command**

.. code-block:: bash

   python scripts/05_infer.py \
     --outputdir outputs/inference \
     --checkpoint runs/pairvae/checkpoints/last.ckpt \
     --data_path data/pair_all_data.h5 \
     --mode les_to_saxs \
     --sample_frac 0.25 \
     --plot

**Arguments**

* ``--outputdir`` – directory where reconstructed arrays, CSV summaries, and
  optional plots are stored. It will be created if missing.
* ``--checkpoint`` – path to the ``.ckpt`` file saved during training.
* ``--data_path`` – evaluation dataset. Use the HDF5 file for VAE models or the
  paired HDF5 for PairVAE. You may also provide a metadata CSV when combined with
  ``--data_dir``.
* ``--mode`` – required when the checkpoint corresponds to a PairVAE. Choose the
  direction to evaluate (e.g. ``les_to_saxs``).
* ``--sample_frac`` – optional fraction (between 0 and 1) of the dataset used
  during inference to save time.
* ``--conversion_dict`` – optional path to the metadata mapping when working
  with categorical labels.
* ``--batch_size`` – number of samples processed per GPU/CPU batch. Increase it
  cautiously to avoid out-of-memory errors.

**Outputs**

* ``outputs/inference/`` – contains reconstructed tensors, generated CSV files,
  and optional ``.png`` plots when ``--plot`` is enabled.
* Console messages confirming the model type and listing the saved artefacts.

.. tip::

   * When evaluating from raw CSVs, use ``--data_dir`` to point to the folder with
     the original ``.txt`` files.
   * If you request ``--plot`` but see no figures, make sure Matplotlib is
     installed in your environment.

Step 5 – Compute validation metrics and SASFit/LES analysis
-----------------------------------------------------------

**Why this matters:** summarises model quality and optionally runs SASFit to
recover physical parameters.

**Command**

.. code-block:: bash

   python scripts/06_val_metrics.py \
     --checkpoint runs/vae/checkpoints/best.ckpt \
     --data_path data/all_data.h5 \
     --conversion_dict data/metadata_dict.json \
     --outputdir outputs/vae_metrics \
     --eval_percentage 0.10 \
     --fit_percentage 0.005
     --mode les_to_saxs

**Arguments**

* ``--checkpoint`` – checkpoint produced in Step 3 or Step 5.
* ``--data_path`` – HDF5 dataset used for evaluation.
* ``--outputdir`` – folder where metric summaries and fit details are
  written.
* ``--mode`` – required for PairVAE checkpoints; choose ``les_to_saxs``,
  ``saxs_to_saxs``, ``les_to_les`` or ``saxs_to_les`` to control the
  evaluation direction. Reconstruction metrics are only produced when the
  input and output domains match (``les_to_les`` ou ``saxs_to_saxs``).
* ``--conversion_dict`` – same JSON produced during conversion; required to
  decode categorical labels.
* ``--eval_percentage`` – fraction of samples evaluated when computing MAE, MSE,
  RMSE, and R² (``0.10`` = 10%).
* ``--fit_percentage`` – fraction of samples passed to the physical fit (SASFit
  ou LES). Keep it small because curve fitting is slow.
* ``--n_processes`` – optional cap on the number of parallel workers. By
  default, the script utilise ``joblib.Parallel`` avec ``n_cpu - 1``.
* ``--random_state`` – ensures reproducible sampling between runs.

**Outputs**

* ``validation_metrics.yaml`` – machine-readable metrics alongside the training
  hyperparameters et l’état de chaque étape.
* ``metrics_summary.txt`` – human-readable recap of the evaluation, incluant le
  statut des reconstructions et le nombre d’échecs lors des fits.
* ``reconstruction_metrics_detailed.csv`` – per-sample scores when
  reconstructions are available.
* ``fit_detailed_results.csv`` – fitted parameters and absolute errors for each
  processed sample.

.. tip::

   * Si SASFit n’est pas installé, lancez le script avec ``--fit_percentage 0``
     pour ignorer les ajustements physiques.
   * Lower ``--eval_percentage`` if you are prototyping and want faster feedback.

(Optional) Run a grid search
-------------------------------------

**Why this matters:** sweeps multiple hyper-parameter combinations to discover
stronger configurations automatically.

**Command**

.. code-block:: bash

   python scripts/03_train.py \
     --mode vae \
     --gridsearch \
     --config configs/vae_grid.yml

**Arguments**

* ``--gridsearch`` – toggles grid-search mode. The script reads search spaces
  from the YAML file and launches multiple experiments.
* ``--config`` – YAML file defining both the baseline configuration and the
  parameter ranges to explore.
* All other arguments described in Step 3 remain available to override defaults.

**Outputs**

* A directory per trial under the location configured in the YAML file, each
  containing checkpoints and metric summaries.
* A console table summarising the hyper-parameter combinations that were
  tried.

.. tip::

   * Start with small ranges to keep the number of experiments manageable.
   * Check the generated log files to identify the best trial and replicate it
     using the single-run command from Step 3 or Step 5.
