AutoFill documentation
======================

Welcome to the AutoFill documentation. This site provides comprehensive guides
for training variational autoencoders on scattering data, including both
single-modality VAEs and paired models for cross-modal reconstruction.

Project overview
----------------

AutoFill automates the data pipeline that supports the research project aiming
to auto-complete scattering experiments such as SAXS and LES. The repository
bundles the CSV preprocessing utilities, the TXT-to-HDF5 converters, and the
training scripts that produce both single-modality VAEs and the paired model
that can translate LES spectra into SAXS reconstructions (and vice versa). The
``src/dataset`` modules expose the modality-specific preprocessing steps, while
``src/model/pairvae`` combines the trained VAEs to enable cross-domain
reconstruction and latent alignment.

Quick start
-----------

1. **Getting started** - Installation and basic setup
2. **Tutorials** - Step-by-step workflows for training and inference
3. **Configuration guide** - Complete parameter reference for training experiments
4. **Data formats** - Understanding HDF5 files and data structures

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   data_formats
   configuration
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
