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
* ``--no_progressbar`` – disable progress bars for cleaner logs (useful for batch jobs and redirected output).

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
* ``--no_progressbar`` – disable progress bars for cleaner logs (useful for batch jobs and redirected output).
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

.. note::
   When training the PairVAE, the PairDataset's ``data_q`` entries are **not** used.
   Instead, the PairVAE loads the ``data_q`` that were produced/used by the single VAEs
   previously trained and saved in their checkpoints (ckpt).


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
   For detailed information about training outputs, directory structure, and monitoring options, see the training outputs section in Step 2.

Step 4 – Run inference
----------------------

**Why this matters:** generates reconstructions or translations from a trained
checkpoint and saves the results for inspection.

**Basic command (VAE)**

.. code-block:: bash

   python scripts/05_infer.py \
     --outputdir outputs/vae_inference \
     --checkpoint DATA/best_saxs.ckpt \
     --data_path DATA/cylinder_saxs_library_no_noise_meta_diameter_metric.h5 \
     --format h5 \
     --plot \
     --plot_limit 20

**Basic command (PairVAE)**

.. code-block:: bash

   python scripts/05_infer.py \
     --outputdir outputs/les_to_saxs \
     --checkpoint DATA/best_pair+.ckpt \
     --data_path DATA/cylinder_les_meta.h5 \
     --mode les_to_saxs \
     --sample_frac 100 \
     --format txt \
     --plot \
     --plot_limit 20 \
     --no_progressbar

**Arguments**

Required arguments:

* ``-o``, ``--outputdir`` – directory where reconstructed arrays and plots are stored. Created automatically if missing.
* ``-c``, ``--checkpoint`` – path to the ``.ckpt`` file saved during training.
* ``-d``, ``--data_path`` – evaluation dataset. Use the HDF5 file for VAE models or the paired HDF5 for PairVAE. You may also provide a metadata CSV when combined with ``--data_dir``.

PairVAE-specific arguments:

* ``--mode`` – **required for PairVAE**. Translation mode to run: ``les_to_saxs``, ``saxs_to_les``, ``les_to_les``, or ``saxs_to_saxs``.

Optional arguments:

* ``-s``, ``--sample_frac`` – fraction or absolute count of the dataset to use:

  * If ``< 1.0``: random fraction (e.g., ``0.25`` = 25% of data)
  * If ``= 1.0``: full dataset (default)
  * If ``> 1.0``: absolute number of samples (e.g., ``100`` = exactly 100 samples)

* ``--format`` – output format for predictions: ``txt`` (individual files) or ``h5`` (single HDF5 file). Default: ``txt``.
* ``--plot`` – save reconstruction plots alongside data outputs.
* ``--plot_limit`` – maximum number of plots to save when ``--plot`` is enabled. Default: ``10``.
* ``-bs``, ``--batch_size`` – number of samples processed per GPU/CPU batch. Default: ``32``. Increase cautiously to avoid out-of-memory errors.
* ``--sample_seed`` – random seed for reproducible sampling. Default: ``1``.
* ``--n_jobs_io`` – number of parallel workers for writing TXT files. Default: ``8``. Higher values speed up I/O for large datasets.
* ``-p``, ``--no_progressbar`` – disable progress bars for cleaner logs (useful for batch jobs and redirected output).
* ``-cd``, ``--conversion_dict`` – path to the metadata conversion dictionary if working with categorical labels.
* ``-dd``, ``--data_dir`` – directory containing raw TXT files when ``--data_path`` points to a CSV file.

**Examples**

Full dataset inference with HDF5 output:

.. code-block:: bash

   python scripts/05_infer.py \
     --outputdir outputs/full_inference \
     --checkpoint runs/SAXS/best.ckpt \
     --data_path DATA/test_data.h5 \
     --format h5 \
     --plot

Random 25% subset with TXT output:

.. code-block:: bash

   python scripts/05_infer.py \
     --outputdir outputs/sample_inference \
     --checkpoint runs/SAXS/best.ckpt \
     --data_path DATA/test_data.h5 \
     --sample_frac 0.25 \
     --format txt \
     --sample_seed 42

Exactly 100 samples for quick validation:

.. code-block:: bash

   python scripts/05_infer.py \
     --outputdir outputs/quick_check \
     --checkpoint runs/SAXS/best.ckpt \
     --data_path DATA/test_data.h5 \
     --sample_frac 100 \
     --plot \
     --plot_limit 5

PairVAE translation with no progress bars (for batch scripts):

.. code-block:: bash

   python scripts/05_infer.py \
     --outputdir outputs/les_to_saxs \
     --checkpoint runs/PAIR/best.ckpt \
     --data_path DATA/pair_data.h5 \
     --mode les_to_saxs \
     --format h5 \
     --no_progressbar

**Outputs**

Depending on the ``--format`` argument:

**HDF5 format** (``--format h5``):

* ``predictions_{mode}.h5`` – single file containing:

  * ``predictions``: inverted model predictions (N, signal_length)
  * ``q``: q-values (N, q_length)
  * ``latents``: latent representations (N, latent_dim) - VAE only
  * ``indices``: sample indices used

**TXT format** (``--format txt``):

* ``predictions_{mode}/`` – directory with individual files:

  * ``prediction_000000.txt`` – q-y paired columns for each sample
  * Parallel I/O controlled by ``--n_jobs_io``

**Plots** (when ``--plot`` is enabled):

* ``plots_{mode}/`` – visualization directory:

  * ``i000000_{mode}.png`` – reconstruction or translation plots
  * Limited by ``--plot_limit`` to avoid generating thousands of files

Console output includes:

* Model type detection (VAE or PairVAE)
* Data transformations being applied
* Progress bars (unless ``--no_progressbar`` is set)
* Output file paths

.. tip::

   * Use ``--format h5`` for large-scale inference (faster I/O, single file)
   * Use ``--format txt`` when you need to integrate with external tools
   * Use ``--no_progressbar`` in SLURM jobs or when redirecting output to files
   * Use ``--sample_frac 0.1`` during development for quick iterations
   * Use ``--sample_seed`` for reproducible sampling across runs
   * Increase ``--n_jobs_io`` (e.g., 16) for faster TXT writing on systems with many cores

.. note::
   The inference script automatically detects:

   * Model type (VAE or PairVAE) from the checkpoint
   * Data modality (SAXS or LES) from model configuration
   * Required transformations from checkpoint metadata
   * Appropriate plot scaling (log-log for SAXS, linear for LES)

Step 5 – Compute validation metrics and SASFit/LES analysis
-----------------------------------------------------------

**Why this matters:** summarizes model quality and optionally runs SASFit to
recover physical parameters.

**Basic command (VAE)**

.. code-block:: bash

   python scripts/06_val_metrics.py \
     --checkpoint DATA/best_saxs.ckpt \
     --data_path DATA/cylinder_saxs_library_no_noise_meta_diameter_metric.h5 \
     --outputdir outputs/vae_metrics \
     --eval_percentage 0.10 \
     --fit_percentage 0.005

**Basic command (PairVAE)**

.. code-block:: bash

   python scripts/06_val_metrics.py \
     --checkpoint DATA/best_pair.ckpt \
     --data_path DATA/pair_data.h5 \
     --outputdir outputs/pair_metrics \
     --mode saxs_to_saxs \
     --eval_percentage 0.10 \
     --fit_percentage 0.005 \
     --no_progressbar

**Arguments**

Required arguments:

* ``-c``, ``--checkpoint`` – path to the checkpoint file produced during training.
* ``-d``, ``--data_path`` – HDF5 dataset used for evaluation.
* ``-o``, ``--outputdir`` – folder where metric summaries and fit details are written.

PairVAE-specific arguments:

* ``--mode`` – required for PairVAE. Evaluation mode: ``les_to_saxs``, ``saxs_to_les``, ``les_to_les``, or ``saxs_to_saxs``. Reconstruction metrics are only computed when input and output domains match (``les_to_les`` or ``saxs_to_saxs``).

Optional arguments:

* ``--eval_percentage`` – fraction of samples to evaluate for reconstruction metrics (MAE, MSE, RMSE, R²). Default: ``0.05`` (5%).
* ``--fit_percentage`` – fraction of samples to pass to physical fitting (SASFit or LES). Default: ``0.05`` (5%). Keep small because curve fitting is slow.
* ``--batch_size`` – inference batch size. Default: ``32``.
* ``--signal_length`` – forced signal length for reconstruction. Default: ``1000``.
* ``--qmin_fit`` – minimum q value for fitting. Default: ``0.001``.
* ``--qmax_fit`` – maximum q value for fitting. Default: ``0.5``.
* ``--factor_scale_to_conc`` – scaling factor to convert to concentration. Default: ``20878``.
* ``--n_processes`` – number of parallel workers for fitting. Default: uses all available CPUs minus 1.
* ``--random_state`` – random seed for reproducible sampling. Default: ``42``.
* ``-p``, ``--no_progressbar`` – disable progress bars for cleaner logs.
* ``-cd``, ``--conversion_dict`` – path to metadata conversion dictionary for categorical labels.

**Examples**

Quick validation with 5% of data:

.. code-block:: bash

   python scripts/06_val_metrics.py \
     --checkpoint runs/SAXS/best.ckpt \
     --data_path DATA/test_data.h5 \
     --outputdir outputs/quick_metrics \
     --eval_percentage 0.05 \
     --fit_percentage 0.01

Full evaluation without physical fitting:

.. code-block:: bash

   python scripts/06_val_metrics.py \
     --checkpoint runs/SAXS/best.ckpt \
     --data_path DATA/test_data.h5 \
     --outputdir outputs/full_metrics \
     --eval_percentage 1.0 \
     --fit_percentage 0

PairVAE reconstruction metrics only:

.. code-block:: bash

   python scripts/06_val_metrics.py \
     --checkpoint runs/PAIR/best.ckpt \
     --data_path DATA/pair_data.h5 \
     --outputdir outputs/pair_reconstruction \
     --mode les_to_les \
     --eval_percentage 0.10 \
     --fit_percentage 0

Batch mode with no progress bars:

.. code-block:: bash

   python scripts/06_val_metrics.py \
     --checkpoint runs/SAXS/best.ckpt \
     --data_path DATA/test_data.h5 \
     --outputdir outputs/batch_metrics \
     --no_progressbar \
     --random_state 42

**Outputs**

The validation script generates the following files in the output directory:

``validation_metrics.yaml``
    Machine-readable metrics including:

    - Training hyperparameters
    - Reconstruction metrics (MAE, MSE, RMSE, R²)
    - Fitting results summary
    - Status of each validation stage

``metrics_summary.txt``
    Human-readable summary including:

    - Model configuration
    - Reconstruction quality metrics
    - Fitting success/failure counts
    - Overall validation status

``reconstruction_metrics_detailed.csv``
    Per-sample reconstruction scores (when applicable):

    - Sample index
    - MAE, MSE, RMSE per sample
    - R² score per sample

``fit_detailed_results.csv``
    Fitted parameters and errors for each sample:

    - Physical parameters recovered by SASFit/LES fitting
    - Absolute errors compared to ground truth
    - Fit quality indicators

Console output includes:

* Validation progress (unless ``--no_progressbar`` is set)
* Metric computation status
* Summary of results
* Output file locations

.. tip::

   * Start with small percentages (5-10%) for quick feedback during development
   * Use ``--fit_percentage 0`` if SASFit is not installed or you only need reconstruction metrics
   * Use ``--no_progressbar`` in SLURM jobs or automated pipelines
   * Set ``--random_state`` for reproducible metrics across runs
   * Increase ``--eval_percentage`` to ``1.0`` for final model evaluation
   * For PairVAE, use matching modes (``les_to_les`` or ``saxs_to_saxs``) to get reconstruction metrics

.. note::
   * Reconstruction metrics (MAE, MSE, RMSE, R²) are only computed when the input and output domains match
   * For PairVAE cross-domain translations (``les_to_saxs`` or ``saxs_to_les``), only fitting metrics are available
   * Physical fitting is computationally expensive; adjust ``--fit_percentage`` based on available time and resources

