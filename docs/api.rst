Datasets
========

HDF5 Datasets
-------------

.. automodule:: src.dataset.datasetH5
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.dataset.datasetPairH5
   :members:
   :undoc-members:
   :show-inheritance:

TXT Dataset
-----------

.. automodule:: src.dataset.datasetTXT
   :members:
   :undoc-members:
   :show-inheritance:

Transformations
---------------

.. automodule:: src.dataset.transformations
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: src.dataset.utils
   :members:
   :undoc-members:
   :show-inheritance:

Single VAE
===========

VAE Submodules
--------------

VAE - Variational Autoencoder
*****************************

.. automodule:: src.model.vae.submodel.VAE
   :synopsis: Implementation of a variational autoencoder (VAE) in PyTorch.
   :members:
   :undoc-members:
   :show-inheritance:

Model
^^^^^

- **Encoder:**  
  Sequential 1D convolutions with downsampling (`stride=2`) → batch normalization → GELU activations.  
  Produces a compressed feature map that is flattened and mapped to latent mean (`μ`) and log-variance (`logσ²`) through fully connected layers.

- **Latent space:**  
  Samples latent vectors `z` via reparameterization.

- **Decoder:**  
  Starts with a linear projection from latent space back to the encoder’s flattened space, followed by symmetric `ConvTranspose1d` layers that progressively upsample to the original resolution.

- **Output layer:**  
  Final transposed convolution followed by a sigmoid activation to produce outputs in the range `[0, 1]`.


Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from src.model.vae.submodel.VAE import VAE
   import torch

   model = VAE(input_dim=1000, latent_dim=64)
   x = torch.randn(8, 1, 1000)  # batch of 8 signals
   output = model(x)

   print(output["recon"].shape)  # (8, 1, 1000)


Notes
^^^^^

- The model automatically determines the compressed size of the latent feature map based on the input length.  
- The final output is padded or cropped if needed to match the original input size.  
- This architecture is suitable for tasks such as: \
   - 1D signal compression  \
   - Denoising  \
   - Generative modeling of sequential data  

ResVAE — Residual Variational Autoencoder
*****************************************


A fully residual **variational autoencoder** designed for 1D signals.  
The encoder is composed of multiple :class:`ResidualBlock` layers followed by dense layers to produce the latent mean and log-variance.  
The decoder reconstructs the signal from the latent space using :class:`ResidualUpBlock` layers.


.. automodule:: src.model.vae.submodel.ResVAE
   :members:
   :undoc-members:
   :show-inheritance:


Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from src.model.vae.submodel.ResVAE import ResVAE
   import torch

   model = ResVAE(input_dim=256, latent_dim=32)
   x = torch.randn(4, 1, 256)  # batch of 4 signals
   output = model(x)

   print(output["recon"].shape)  # (4, 1, 256)


Notes
^^^^^

- Residual connections help stabilize training and mitigate vanishing gradients.  
- Handles non-even input lengths (adjusts `output_padding` dynamically).  
- Suitable for tasks such as compression, denoising, or generative modeling of 1D data.


Lightning VAE
-------------

.. automodule:: src.model.vae.pl_vae
   :members:
   :undoc-members:
   :show-inheritance:

PairVAE
=======

PairVAE 
-------

The PairVAE we implemented is from the article :

    Pair-Variational Autoencoders for Linking and Cross-Reconstruction of Characterization Data from Complementary Structural Characterization Techniques
    Shizhao Lu and Arthi Jayaraman

.. automodule:: src.model.pairvae.pairvae
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: src.model.pairvae.loss
   :members:
   :undoc-members:
   :show-inheritance:

Lightning PairVAE
-----------------

.. automodule:: src.model.pairvae.pl_pairvae
   :members:
   :undoc-members:
   :show-inheritance:


Training and Experiment Control
================================

Train Pipeline
--------------

.. automodule:: src.model.trainer
   :members:
   :undoc-members:
   :show-inheritance:

Grid Search
-----------

.. automodule:: src.model.grid_search
   :members:
   :undoc-members:
   :show-inheritance:



Callbacks
=========

.. automodule:: src.model.callbacks.metrics_callback
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.model.callbacks.inference_callback
   :members:
   :undoc-members:
   :show-inheritance:

