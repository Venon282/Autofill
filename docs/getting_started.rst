Getting started
===============

.. note::
   **About AutoFill**

   AutoFill is a deep learning toolbox for the generation of nanoparticules signals. This toolkit enables training of variational autoencoders (VAEs) on SAXS and LES data, including advanced cross-modal reconstruction capabilities through PairVAE models. Whether you're working with single-modality data or paired measurements, AutoFill streamlines the entire pipeline from data preprocessing to model training and inference.

   **Authors:**

   * **Julien Rabault** - julien.rabault@irit.fr
   * **Caroline de Pourtalès** - caroline.de-pourtales@irit.fr

   Ready to get started? Follow the workflow below to begin using AutoFill for your scattering data analysis.

This guide will help you get started with AutoFill, from installation to running your first training experiment.

Installation
------------

.. note::
   **GPU and CUDA Requirements**

   AutoFill is designed for machine learning workloads and it is **strongly recommended** to have a GPU with CUDA support for optimal performance. Training VAE models on CPU can be extremely slow. For GPU setup instructions and CUDA installation, see the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

**Option 1: Using pip (traditional method)**

1. **Clone the repository and install dependencies**:

   .. code-block:: bash

      git clone https://gitlab.irit.fr/pnria/exterieure/AutoFill.git

      cd AutoFill

      pip install -r requirements.txt

**Option 2: Using uv (recommended for faster installation)**

1. **Clone the repository and install dependencies with uv**:

   .. code-block:: bash

      git clone https://gitlab.irit.fr/pnria/exterieure/AutoFill.git

      cd AutoFill

      uv sync
      # This will create a virtual environment and install all dependencies

   .. tip::
      ``uv`` is a fast Python package installer that can significantly speed up dependency installation. If you don't have ``uv`` installed, you can install it with ``pip install uv`` or follow the installation guide at `uv docs <https://github.com/astral-sh/uv>`_

   .. tip::
      You can use ``uv run <script>`` to run Python scripts within the uv environment without activating it explicitly. See `uv run docs <https://docs.astral.sh/uv/guides/scripts/#running-a-script-with-dependencies>`_

Quick start workflow
--------------------

The typical AutoFill workflow consists of four main steps:

1. **Data preparation**

   - Preprocess your CSV metadata files
   - Convert TXT files to HDF5 format
   - See :doc:`data_formats` for detailed file structure information

2. **Configuration setup**

   - Create or modify YAML configuration files
   - Set model parameters, training settings, and data paths
   - **Important**: See :doc:`configuration` for complete parameter reference and examples

3. **Training**

   - Run training with your configuration file
   - Monitor progress with TensorBoard or MLFlow
   - **Configuration required**: All training parameters must be specified in YAML files

4. **Inference and evaluation**

   - Use trained models for reconstruction or cross-modal translation
   - Evaluate model performance with validation metrics

Documentation structure
-----------------------

- :doc:`configuration` - **Essential reference** for all training parameters
- :doc:`data_formats` - HDF5 file structures and data organization
- :doc:`tutorials` - Step-by-step workflows with command examples
- :doc:`api/index` - API reference for developers

Next steps
----------

1. **Read the configuration guide**: :doc:`configuration` contains all parameters you need for training
2. **Understand data formats**: :doc:`data_formats` explains the HDF5 structure
3. **Follow tutorials**: :doc:`tutorials` provides step-by-step examples
4. **Check API reference**: :doc:`api/index` for detailed module documentation
