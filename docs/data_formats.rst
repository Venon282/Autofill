Data Formats
============

This section describes the data formats used throughout the AutoFill pipeline,
including HDF5 file structures, JSON metadata dictionaries, and CSV layouts.

.. contents::
   :local:
   :depth: 2

HDF5 Data Files
---------------

The AutoFill pipeline uses HDF5 files to store preprocessed time series data
along with associated metadata. These files are produced by the conversion
scripts and consumed by the training and inference utilities.

VAE Data Format (all_data.h5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scripts/02_txtTOhdf5.py`` script produces a single HDF5 file that packs
the cleaned metadata and the padded time series. The layout is intentionally
simple so that it can be explored with tools such as ``h5ls`` or ``h5py``.

**File Structure:**

.. code-block:: none

   data/all_data.h5
   ├── data_q          [N × pad_size] float64
   ├── data_y          [N × pad_size] float64
   ├── len             [N] int64
   ├── csv_index       [N] int64
   └── <metadata_cols> [N] float64 or int64

**Dataset Descriptions:**

``data_q``
    Two-dimensional dataset of shape ``[N, pad_size]`` containing the q-axis
    for each sample. Values are stored as ``float64``. Each row represents
    one time series with padding applied to reach the target length.

``data_y``
    Two-dimensional dataset of shape ``[N, pad_size]`` with the signal values
    aligned with ``data_q``. Values are stored as ``float64``. Padding values
    are typically zeros.

``len``
    One-dimensional dataset ``[N]`` storing the original length of each time
    series before padding or truncation. This allows you to identify the
    actual data points vs. padding.

``csv_index``
    One-dimensional dataset ``[N]`` linking each sample back to its row in
    the metadata CSV. Use this index to cross-reference sample properties
    with the original CSV file.

``<metadata columns>``
    For every non-excluded column from the CSV, a dataset of shape ``[N]``
    is added with the same name as the original column. Data types depend
    on the column content:

    * **Numeric columns**: Stored as ``float64`` directly
    * **String columns**: Mapped to integers using the conversion dictionary
      (see :ref:`json-metadata-dict`)
    * **Missing values**: Stored as ``-1``

**Example Usage:**

.. code-block:: python

   import h5py

   with h5py.File('data/all_data.h5', 'r') as f:
       print("Available datasets:", list(f.keys()))

       # Load first 5 samples
       q_data = f['data_q'][:5]
       y_data = f['data_y'][:5]
       lengths = f['len'][:5]

       print(f"Q-axis shape: {q_data.shape}")
       print(f"Original lengths: {lengths}")

PairVAE Data Format (pair_all_data.h5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scripts/04_pair_txtTOhdf5.py`` script creates HDF5 files for paired
modality training. The structure extends the single-modality format to
accommodate two related datasets.

**File Structure:**

.. code-block:: none

   data/pair_all_data.h5
   ├── data_q_mod1     [N × pad_size] float64
   ├── data_y_mod1     [N × pad_size] float64
   ├── data_q_mod2     [N × pad_size] float64
   ├── data_y_mod2     [N × pad_size] float64
   ├── len_mod1        [N] int64
   ├── len_mod2        [N] int64
   ├── csv_index       [N] int64
   └── <metadata_cols> [N] float64 or int64

**Dataset Descriptions:**

The paired format follows the same conventions as the single-modality version
but duplicates the data arrays for each modality:

``data_q_mod1``, ``data_y_mod1``
    Q-axis and signal data for the first modality (e.g., LES data)

``data_q_mod2``, ``data_y_mod2``
    Q-axis and signal data for the second modality (e.g., SAXS data)

``len_mod1``, ``len_mod2``
    Original lengths before padding for each modality

The remaining datasets (``csv_index`` and metadata columns) work identically
to the single-modality case.

.. _json-metadata-dict:

JSON Metadata Dictionary
------------------------

The conversion scripts generate JSON files that map categorical string values
to their integer encodings used in the HDF5 datasets.

**File Structure:**

.. code-block:: json

   {
     "column_name_1": {
       "category_a": 0,
       "category_b": 1,
       "category_c": 2
     },
     "column_name_2": {
       "low": 0,
       "medium": 1,
       "high": 2
     }
   }

**Usage Example:**

.. code-block:: python

   import json
   import h5py

   # Load the conversion dictionary
   with open('data/metadata_dict.json', 'r') as f:
       conv_dict = json.load(f)

   # Load encoded values from HDF5
   with h5py.File('data/all_data.h5', 'r') as f:
       material_encoded = f['material'][:]

   # Convert back to original strings
   material_map = {v: k for k, v in conv_dict['material'].items()}
   material_decoded = [material_map.get(x, 'unknown') for x in material_encoded]

CSV Data Format
---------------

Input CSV files should follow standard comma-separated format with a header
row. The preprocessing script (``01_csv_pre_process.py``) handles different
separators and normalizes the output.

**Required Columns:**

``path``
    Absolute or relative path to the corresponding ``.txt`` time series file.
    This column is used to locate and load the raw data during conversion.

**Optional Columns:**

Any additional columns will be included in the HDF5 metadata datasets. Common
examples include:

* ``material``: Sample material type
* ``concentration``: Measurement concentration
* ``temperature``: Measurement temperature
* ``diameter``: Physical parameter for validation
* ``length``: Physical parameter for validation

**Missing Values:**

Missing or invalid entries should be left empty or marked as ``NaN``. The
conversion process maps these to ``-1`` in the final HDF5 datasets.

**Example CSV:**

.. code-block:: none

   path,material,concentration,diameter,temperature
   data/txt/sample001.txt,silver,0.1,25.3,298
   data/txt/sample002.txt,gold,0.05,30.1,300
   data/txt/sample003.txt,silver,,28.7,295

Validation and Debugging
------------------------

Before training or inference, validate your data files using the provided
utilities:

**Check File Integrity:**

.. code-block:: bash

   python scripts/saminitycheck.py data/metadata_clean.csv

**Inspect HDF5 Contents:**

.. code-block:: bash

   h5ls -r data/all_data.h5
   h5dump -H data/all_data.h5

**Quick Python Check:**

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('data/all_data.h5', 'r') as f:
       print(f"Total samples: {f['data_q'].shape[0]}")
       print(f"Pad size: {f['data_q'].shape[1]}")
       print(f"Average length: {np.mean(f['len'][:]):.1f}")
       print(f"Length range: {np.min(f['len'][:])}-{np.max(f['len'][:])}")
