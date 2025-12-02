Configuration
=============

.. note::
   **Important**: This documentation reflects the current Pydantic-based configuration system.
   All configurations are validated at runtime with automatic type checking and helpful error messages.

.. contents::
   :local:
   :depth: 2

Quick Start
-----------

Minimal VAE configuration:

.. code-block:: yaml

   experiment_name: "my_saxs_vae"
   run_name: "baseline"

   model:
     type: vae
     spec: saxs
     vae_class: ResVAE
     beta: 1.0e-7
     args:
       in_channels: 1
       output_channels: 1
       input_dim: 1000
       latent_dim: 64
       down_channels: [8, 16, 32, 64, 128, 256, 512]
       up_channels: [512, 256, 128, 64, 32, 16, 8]

   dataset:
     hdf5_file: DATA/cylinder_saxs_library.h5
     transforms_data:
       y:
         PreprocessingSAXS:
           pad_size: 1000
       q:
         PreprocessingQ:
           pad_size: 1000

   training:
     output_dir: runs/
     num_epochs: 300
     batch_size: 8
     max_lr: 1.0e-04
     patience: 40

Configuration Structure
-----------------------

Pydantic Validation
~~~~~~~~~~~~~~~~~~~

AutoFill uses **Pydantic** models for configuration validation:

- **Automatic type checking**: Fields are validated against their declared types
- **Range validation**: Numeric fields check min/max constraints (e.g., ``batch_size: int >= 1``)
- **Default values**: Unspecified fields use sensible defaults with warnings
- **Clear error messages**: Invalid configurations produce detailed error reports

Example validation error:

.. code-block:: text

   ValidationError: 2 validation errors for VAETrainingConfig
   batch_size
     Input should be greater than or equal to 1 [type=greater_than_equal]
   max_lr
     Input should be greater than 0 [type=greater_than]

Top-Level Fields
~~~~~~~~~~~~~~~~

``experiment_name`` (string, required)
    Name of the experiment group. Used for organizing runs in output directories and MLFlow/TensorBoard.

    Example: ``"AUTOFILL_SAXS"``

``run_name`` (string, required)
    Unique identifier for this specific run. Combined with ``experiment_name`` to create output paths.

    Example: ``"saxs_cylinder_ag"``

``mlflow_uri`` (string, optional)
    MLFlow tracking server URI. If omitted, uses TensorBoard logging.

    Example: ``"https://mlflowts.irit.fr/"``

Model Configuration
-------------------

VAE Models
~~~~~~~~~~

Configuration class: ``VAEModelConfig`` (inherits from ``BaseModelConfig``)

.. code-block:: yaml

   model:
     type: vae                    # ModelType.VAE
     spec: saxs                   # ModelSpec.SAXS or ModelSpec.LES
     vae_class: ResVAE            # Registered architecture name
     beta: 1.0e-7                 # KL divergence weight
     args:                        # Architecture-specific arguments
       in_channels: 1
       output_channels: 1
       input_dim: 1000
       latent_dim: 64
       use_sigmoid: False
       down_channels: [8, 16, 32, 64, 128, 256, 512]
       up_channels: [512, 256, 128, 64, 32, 16, 8]

**Required fields:**

``type`` (ModelType enum)
    Must be ``"vae"`` for single-modality VAE.

    - Can be overridden with ``--mode vae`` command-line argument

``spec`` (ModelSpec enum)
    Data modality specification. Options: ``"saxs"``, ``"les"``

    - Determines default transformations and plot scaling

``vae_class`` (string)
    Registered VAE architecture class name. Available options:

    - ``"ResVAE"`` - Residual VAE with skip connections (recommended)
    - Check ``src/model/vae/submodel/registry.py`` for available models

``args`` (dict)
    Architecture-specific constructor arguments. **Varies by vae_class**.

**Optional fields:**

``beta`` (float, default: 1.0e-7)
    β-VAE KL divergence scaling coefficient. Range: ``>= 0.0``

    - Higher values enforce stronger disentanglement
    - Lower values prioritize reconstruction quality

``data_q`` (array, optional)
    Q-values array. Usually loaded from checkpoint, not needed in config.

``transforms_data`` (dict, optional)
    Stored in checkpoint after training. See Dataset Configuration.

``verbose`` (bool, default: True)
    Enable configuration validation warnings.

**ResVAE architecture arguments:**

The ``args`` dict for ResVAE accepts:

- ``in_channels`` (int): Input channels (typically 1 for spectra)
- ``output_channels`` (int): Output channels (typically 1)
- ``input_dim`` (int): Length of input spectrum (e.g., 1000)
- ``latent_dim`` (int): Latent space dimensionality (e.g., 64, 128)
- ``use_sigmoid`` (bool): Apply sigmoid to output (typically False)
- ``down_channels`` (list[int]): Encoder channel progression
- ``up_channels`` (list[int]): Decoder channel progression (typically reversed)

Example configurations:

.. code-block:: yaml

   # Small model (fast, less capacity)
   args:
     latent_dim: 32
     down_channels: [8, 16, 32, 64]
     up_channels: [64, 32, 16, 8]

   # Large model (slow, high capacity)
   args:
     latent_dim: 128
     down_channels: [16, 32, 64, 128, 256, 512]
     up_channels: [512, 256, 128, 64, 32, 16]

PairVAE Models
~~~~~~~~~~~~~~

Configuration class: ``PairVAEModelConfig`` (inherits from ``BaseModelConfig``)

.. code-block:: yaml

   model:
     type: pair_vae
     ckpt_path_saxs: runs/SAXS/saxs_baseline/best.ckpt
     ckpt_path_les: runs/LES/les_baseline/best.ckpt
     freeze_subvae: False

**Required fields:**

``type`` (ModelType enum)
    Must be ``"pair_vae"`` for paired-modality VAE.

``ckpt_path_saxs`` (string, optional but recommended)
    Path to pretrained SAXS VAE checkpoint (``.ckpt`` file).

    - If not provided, SAXS encoder/decoder are randomly initialized
    - Checkpoint must contain a trained VAE model with spec=saxs

``ckpt_path_les`` (string, optional but recommended)
    Path to pretrained LES VAE checkpoint (``.ckpt`` file).

    - If not provided, LES encoder/decoder are randomly initialized
    - Checkpoint must contain a trained VAE model with spec=les

**Optional fields:**

``freeze_subvae`` (bool, default: False)
    Freeze pretrained VAE weights during PairVAE training.

    - ``True``: Only train cross-modal connections (faster, preserves pretrained quality)
    - ``False``: Fine-tune entire model (slower, may improve alignment)

.. important::
   **Checkpoint requirements:**

   The checkpoint files (``*.ckpt``) must be complete PyTorch Lightning checkpoints containing:

   - Model state dict
   - ``model_config`` with VAE configuration
   - ``train_config`` with training settings
   - ``transforms_data`` with transformation pipelines
   - ``data_q`` array

   These are automatically saved by AutoFill training runs.

Dataset Configuration
---------------------

Single-Modality HDF5 Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration class: ``HDF5DatasetConfig`` (inherits from ``BaseDatasetConfig``)

.. code-block:: yaml

   dataset:
     hdf5_file: DATA/cylinder_saxs_library_no_noise_meta_diameter_metric.h5
     conversion_dict: null
     metadata_filters: null
     requested_metadata: null
     use_data_q: False
     transforms_data:
       q:
         PreprocessingQ:
           pad_size: 1000
       y:
         PreprocessingSAXS:
           pad_size: 1000

**Required fields:**

``hdf5_file`` (string or Path)
    Path to HDF5 dataset file. **Note**: Field name is ``hdf5_file``, not ``h5_file_path``.

    - Can be overridden with ``--hdf5_file`` command-line argument
    - Must contain required datasets: ``data_q``, ``data_y``

**Optional fields:**

``conversion_dict`` (dict, string, or Path, optional)
    Metadata conversion dictionary for categorical variables.

    - Can be a dict, path to JSON file, or null
    - Maps categorical labels to numeric codes

``metadata_filters`` (dict, optional)
    Filters to select specific samples based on metadata.

    Example:

    .. code-block:: yaml

       metadata_filters:
         material: [0, 1]  # Keep only materials 0 and 1
         diameter_nm: [10, 20, 30]  # Keep specific diameters

``requested_metadata`` (list[str], optional)
    Metadata columns to load. If null, loads all available metadata.

``use_data_q`` (bool, default: False)
    Include q-values in batch outputs.

``transforms_data`` (dict, optional)
    Transformation pipelines for q and y data.

    Structure:

    .. code-block:: yaml

       transforms_data:
         q:  # Q-axis transformations
           TransformerName:
             param1: value1
         y:  # Signal transformations
           TransformerName:
             param1: value1

``verbose`` (bool, default: True)
    Enable dataset validation warnings.

**Available transformations:**

For SAXS data (``y``):

.. code-block:: yaml

   transforms_data:
     y:
       PreprocessingSAXS:
         pad_size: 1000  # Pad/truncate to fixed length
         value: 0        # Padding value

Preprocessing pipeline: Padding → EnsurePositive → Log → MinMaxScaler

For LES data (``y``):

.. code-block:: yaml

   transforms_data:
     y:
       PreprocessingLES:
         pad_size: 500
         value: 0

Preprocessing pipeline: Padding → MinMaxScaler

For Q-axis (``q``):

.. code-block:: yaml

   transforms_data:
     q:
       PreprocessingQ:
         pad_size: 1000
         value: 0

Preprocessing pipeline: Padding only

Paired-Modality HDF5 Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration class: ``PairHDF5DatasetConfig`` (inherits from ``BaseDatasetConfig``)

.. code-block:: yaml

   dataset:
     hdf5_file: DATA/pair_data.h5
     conversion_dict: null
     metadata_filters: null
     requested_metadata: null

**Required fields:**

``hdf5_file`` (string or Path)
    Path to paired HDF5 dataset.

    Must contain datasets:

    - ``data_q_saxs``, ``data_y_saxs``
    - ``data_q_les``, ``data_y_les``

.. note::
   Transformations for PairVAE are loaded from the pretrained VAE checkpoints,
   **not** from the dataset configuration. The transforms stored in
   ``ckpt_path_saxs`` and ``ckpt_path_les`` are used automatically.

Training Configuration
----------------------

Base Training Settings
~~~~~~~~~~~~~~~~~~~~~~~

Configuration class: ``VAETrainingConfig`` or ``PairVAETrainingConfig`` (inherit from ``BaseTrainingConfig``)

.. code-block:: yaml

   training:
     output_dir: runs/
     num_epochs: 300
     batch_size: 8
     max_lr: 1.0e-04
     patience: 40
     num_gpus: 1
     num_workers: 1

     # Data splits
     train_indices_path: DATA/train_pairs_saxs_les.npy
     val_indices_path: DATA/val_pairs_saxs_les.npy
     test_indices_path: DATA/test_pairs_saxs_les.npy

     # Learning rate scheduler
     warmup_epochs: 5
     eta_min: 1.0e-15
     min_delta: 0.0000001

     # Inference callbacks
     plot_train: True
     plot_val: True
     use_loglog: True
     num_samples: 10
     every_n_epochs: 10

     # Loss weighting
     weighted_loss: False
     weighted_loss_limit_index: null
     sample_frac: 1.0

**Core training parameters:**

``output_dir`` (string, default: "train_results")
    Base directory for outputs. Final path: ``{output_dir}/{experiment_name}/{run_name}/``

``num_epochs`` (int, default: 300, >= 1)
    Maximum training epochs.

``batch_size`` (int, default: 8, >= 1)
    Batch size for training and validation.

``max_lr`` (float, default: 1.0e-4, > 0)
    Maximum learning rate for optimizer.

``patience`` (int, default: 40, >= 0)
    Early stopping patience (epochs without improvement).

``num_gpus`` (int, default: 1, >= 0)
    Number of GPUs to use. Set to 0 for CPU-only.

``num_workers`` (int, default: 4, >= 0)
    DataLoader workers. Set to 0 for single-threaded loading.

**Data split parameters:**

``train_indices_path`` (string or Path, optional)
    Path to ``.npy`` file with training indices.

    - If provided with ``val_indices_path``, uses precomputed splits
    - If both are null, uses automatic 80/20 split
    - **Important**: Use the same indices for VAE and PairVAE training to prevent data leakage

``val_indices_path`` (string or Path, optional)
    Path to ``.npy`` file with validation indices.

``test_indices_path`` (string or Path, optional)
    Path to ``.npy`` file with test indices.

.. important::
   **Index files must contain:**

   - For single VAE: 1D array of integers (dataset indices)
   - For PairVAE: Array of tuples ``(pair_idx, saxs_idx, les_idx)``

   Load with: ``np.load(path, allow_pickle=True)``

**Learning rate scheduler:**

``warmup_epochs`` (int, default: 5, >= 0)
    Number of epochs for linear warmup.

``eta_min`` (float, default: 1.0e-15, >= 0)
    Minimum learning rate for scheduler.

``min_delta`` (float, default: 1.0e-7, >= 0)
    Minimum change in validation loss for early stopping.

**Callbacks and visualization:**

``plot_train`` (bool, default: True)
    Generate reconstruction plots from training set.

``plot_val`` (bool, default: True)
    Generate reconstruction plots from validation set.

``use_loglog`` (bool, default: True)
    Use log-log scale for plots (appropriate for SAXS).

``num_samples`` (int, default: 10, >= 1)
    Number of samples to plot.

``every_n_epochs`` (int, default: 10, >= 1)
    Plot frequency (in epochs).

``save_every`` (int, default: 1, >= 1)
    Checkpoint saving frequency (in epochs).

**Loss configuration:**

``weighted_loss`` (bool, default: False)
    Enable weighted MSE loss.

``weighted_loss_limit_index`` (int, optional)
    Index where weights change. Points before this index get weight 10.0, after get 1.0.

``sample_frac`` (float, default: 1.0, [0.0, 1.0])
    Fraction of dataset to use (for debugging/fast iteration).

PairVAE-Specific Training
~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration class: ``PairVAETrainingConfig``

Additional fields for PairVAE:

.. code-block:: yaml

   training:
     # ... base training parameters ...

     # Barlow Twins loss
     lambda_param: 0.005

     # Loss component weights
     weight_latent_similarity: 1.0
     weight_saxs2saxs: 1.0
     weight_les2les: 1.0
     weight_saxs2les: 1.0
     weight_les2saxs: 1.0

``lambda_param`` (float, default: 0.005, >= 0)
    Barlow Twins loss coefficient for latent space alignment.

``weight_latent_similarity`` (float, default: 1.0, >= 0)
    Weight for latent space similarity loss.

``weight_saxs2saxs`` (float, default: 1.0, >= 0)
    Weight for SAXS autoencoding reconstruction loss.

``weight_les2les`` (float, default: 1.0, >= 0)
    Weight for LES autoencoding reconstruction loss.

``weight_saxs2les`` (float, default: 1.0, >= 0)
    Weight for SAXS-to-LES translation loss.

``weight_les2saxs`` (float, default: 1.0, >= 0)
    Weight for LES-to-SAXS translation loss.

**Total loss formula:**

.. code-block:: text

   total_loss =
     weight_latent_similarity × latent_loss +
     weight_saxs2saxs × saxs_recon_loss +
     weight_les2les × les_recon_loss +
     weight_saxs2les × saxs_to_les_loss +
     weight_les2saxs × les_to_saxs_loss

Command-Line Overrides
----------------------

The training script (``03_train.py``) accepts command-line arguments that override YAML config values:

.. code-block:: bash

   python scripts/03_train.py \
     --config config/vae_saxs.yaml \
     --mode vae \
     --hdf5_file DATA/custom_data.h5 \
     --name my_custom_run \
     --dry-run

**Available arguments:**

``--config`` (required)
    Path to YAML configuration file.

``--mode`` (optional)
    Override ``model.type``. Options: ``vae``, ``pair_vae``

``--hdf5_file`` (optional)
    Override ``dataset.hdf5_file``.

``--name`` (optional)
    Override ``run_name``.

``--spec`` (optional)
    Override ``model.spec``. Options: ``saxs``, ``les``, ``pair``

``--conversion_dict_path`` (optional)
    Override ``dataset.conversion_dict``.

``--technique`` (optional)
    Add metadata filter for technique.

``--material`` (optional)
    Add metadata filter for material.

``--dry-run``
    Validate configuration without starting training.

``--verbose``
    Enable verbose logging.

``--gridsearch``
    Run grid search using ``param_grid`` section.

Example Complete Configurations
--------------------------------

SAXS VAE (Minimal)
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   experiment_name: AUTOFILL_SAXS
   run_name: saxs_baseline

   model:
     type: vae
     spec: saxs
     vae_class: ResVAE
     beta: 1.0e-7
     args:
       in_channels: 1
       output_channels: 1
       input_dim: 1000
       latent_dim: 64
       down_channels: [8, 16, 32, 64, 128, 256, 512]
       up_channels: [512, 256, 128, 64, 32, 16, 8]

   dataset:
     hdf5_file: DATA/cylinder_saxs_library.h5
     transforms_data:
       y:
         PreprocessingSAXS:
           pad_size: 1000
       q:
         PreprocessingQ:
           pad_size: 1000

   training:
     output_dir: runs/
     num_epochs: 300
     batch_size: 8
     max_lr: 1.0e-04

PairVAE (Complete)
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   experiment_name: MULTIMODAL
   run_name: pairvae_baseline
   mlflow_uri: https://mlflowts.irit.fr/

   model:
     type: pair_vae
     ckpt_path_saxs: runs/SAXS/saxs_baseline/best.ckpt
     ckpt_path_les: runs/LES/les_baseline/best.ckpt
     freeze_subvae: False

   dataset:
     hdf5_file: DATA/pair_data.h5
     train_indices_path: DATA/train_pairs_saxs_les.npy
     val_indices_path: DATA/val_pairs_saxs_les.npy

   training:
     output_dir: runs/
     num_epochs: 200
     batch_size: 8
     max_lr: 1.0e-05
     patience: 20

     lambda_param: 0.005
     weight_latent_similarity: 1.0e-05
     weight_les2les: 1.0
     weight_les2saxs: 1.0
     weight_saxs2les: 1.0
     weight_saxs2saxs: 1.0

Grid Search Configuration
-------------------------

Add a ``param_grid`` section to explore hyperparameters:

.. code-block:: yaml

   param_grid:
     model.args.latent_dim: [32, 64, 128]
     training.max_lr: [1.0e-03, 1.0e-04, 1.0e-05]
     model.beta: [1.0e-05, 1.0e-06, 1.0e-07]

Run with:

.. code-block:: bash

   python scripts/03_train.py --config config.yaml --gridsearch

Troubleshooting
---------------

**ValidationError during config loading:**

Check that all required fields are present and field names match exactly (e.g., ``hdf5_file``, not ``h5_file_path``).

**Checkpoint loading errors:**

Ensure checkpoint files contain all required keys:

- ``model_config``
- ``train_config``
- ``state_dict``
- ``transforms_data``
- ``data_q``

**Transform mismatch warnings:**

PairVAE loads transforms from checkpoints. Ensure the pretrained VAE models were trained with compatible pad sizes.

**Index file errors:**

For PairVAE indices, ensure:

.. code-block:: python

   # Save indices with allow_pickle=True
   np.save('indices.npy', indices_array, allow_pickle=True)

   # Load with allow_pickle=True
   indices = np.load('indices.npy', allow_pickle=True)

