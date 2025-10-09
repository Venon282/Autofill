AutoFill Documentation
======================

Welcome to the AutoFill documentation. This site provides an overview of the
project along with API references for the available datasets and training
utilities.

Project Overview
----------------

AutoFill automates the data pipeline that supports the research project aiming
to auto-complete scattering experiments such as SAXS and LES. The repository
bundles the CSV preprocessing utilities, the TXT-to-HDF5 converters, and the
training scripts that produce both single-modality VAEs and the paired model
that can translate LES spectra into SAXS reconstructions (and vice versa). The
``src/dataset`` modules expose the modality-specific preprocessing steps, while
``src/model/pairvae`` combines the trained VAEs to enable cross-domain
reconstruction and latent alignment.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   data_formats
   tutorials
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
