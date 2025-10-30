Deep Learning - Notes
=====================

VAE models
----------

VAE - Variational Autoencoder
*****************************
.. module:: src.model.vae.submodel.VAE
   :synopsis: Implementation of a variational autoencoder (VAE) in PyTorch.


A standard **1D convolutional VAE** that learns to encode input signals into a low-dimensional latent space and reconstruct them from sampled latent vectors.  
The architecture consists of symmetric convolutional and transposed-convolutional layers with `BatchNorm1d` and `GELU` activations.

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
- This architecture is suitable for tasks such as:
  - 1D signal compression  
  - Denoising  
  - Generative modeling of sequential data  

ResVAE — Residual Variational Autoencoder
*****************************************

.. module:: src.model.vae.submodel.ResVAE
   :synopsis: Implementation of a residual variational autoencoder (VAE) in PyTorch.


This module implements a **Variational Autoencoder (VAE)** architecture built with **1D residual blocks**, suitable for sequential data such as time series or spectra.  
The architecture follows a symmetric encoder–decoder design, where each convolutional or transposed-convolutional layer is enhanced by a **residual (skip) connection** to improve gradient flow and training stability.

Model 
^^^^^

ResidualBlock
~~~~~~~~~~~~~

.. autoclass:: src.model.vae.submodel.ResVAE.ResidualBlock
   :members:
   :undoc-members:
   :show-inheritance:


A 1D convolutional residual block composed of two convolution layers and GELU activations.

- Downsamples the input by a factor of 2 (using `stride=2`).
- Includes a skip connection to preserve information and stabilize training.

ResidualUpBlock
~~~~~~~~~~~~~~~

.. autoclass:: src.model.vae.submodel.ResVAE.ResidualUpBlock
   :members:
   :undoc-members:
   :show-inheritance:


A symmetric counterpart to :class:`ResidualBlock`, using transposed convolutions for upsampling.

- Doubles the spatial resolution via a `ConvTranspose1d`.
- Includes a transposed skip connection to maintain feature consistency.

ResVAE
~~~~~~~

.. autoclass:: src.model.vae.submodel.ResVAE.ResVAE
   :members:
   :undoc-members:
   :show-inheritance:

A fully residual **variational autoencoder** designed for 1D signals.  
The encoder is composed of multiple :class:`ResidualBlock` layers followed by dense layers to produce the latent mean and log-variance.  
The decoder reconstructs the signal from the latent space using :class:`ResidualUpBlock` layers.


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


PairVAE model
-------------

The PairVAE we implemented is from the article :


    Pair-Variational Autoencoders for Linking and Cross-Reconstruction of Characterization Data from Complementary Structural Characterization Techniques
    Shizhao Lu and Arthi Jayaraman


Losses
------

Kullback-Leibler divergence Loss
********************************

The Kullback–Leibler (KL) divergence term in a Variational Autoencoder (VAE) measures how much the learned latent distribution
diverges from the prior (usually a standard normal distribution). 
It encourages the latent space to remain continuous and structured by penalizing deviations from the unit Gaussian.

The KL divergence loss is given by:

.. math::

   \mathcal{L}_{KL} = -\frac{1}{2} \, \mathbb{E}\left[\sum_{i=1}^{d} (1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2)\right]


Reconstruction Loss
*******************

We use the Mean Squared Error (MSE) as the reconstruction metric. This loss measures how close the reconstructed output :math:`\hat{x}` is to the original input :math:`x`, encouraging the decoder to accurately reproduce the input data.

The reconstruction loss is defined as:

.. math::

   \mathcal{L}_{\text{rec}} = \frac{1}{N} \sum_{i=1}^{N} \| x_i - \hat{x}_i \|_2^2

If you wish to use Binary Cross Entropy loss for a test, you can activate the Sigmoid layer in VAE.py or ResVAE.py.

Barlow-Twinn Loss
*****************

The Barlow-Twinn Loss normalizes the D dimensinonal vectors from the projection head and then computes the DxD cross-correlation matrix between the normalized vectors of the 2 views of each signal.

Then it splits this cross-correlation matrix into two parts. The first part, the diagonal of this matrix is brought closer to 1, 
which pushes up the cosine similarity between the latent vectors of two views of each signal, thus making the backbone invariant to the transformations applied to the views. 
The second part of the loss pushes the non-diagonal elements of the cross-corrlelation matrix closes to 0.
This reduces the redundancy between the different dimensions of the latent vector.


.. autoclass:: src.model.pairvae.loss.BarlowTwinsLoss
   :members:
   :undoc-members:
   :show-inheritance:

Problems you may encounter 
--------------------------


Reconstruction Loss not progressing
***********************************

This con be due to the parameter **beta** in the training configuration.
The lower is **beta** the more the network will focus on reconstructing and less on the Kullback-Leibler loss.

Out of Memory Error
*******************

Out of Memory errors can occur when the GPU has not enough RAM to contain the network, the gradients and the data. 
Try lowering the **batch size**.

Training finishing before the number of epochs is reached
*********************************************************

There is an EarlyStopping Callback meaning that if the loss has not progressed in **patience** epcohs of at least **min_delta** the training will stop.

Exploding Gradient, Nans, ect 
*****************************

When training deep generative models such as VAEs or PairVAEs, numerical instabilities like exploding gradients or NaN losses can arise from several sources.

Learning Rate Too High
^^^^^^^^^^^^^^^^^^^^^^^^^
**Symptoms:**
- Gradients explode or the loss becomes NaN.
- Training diverges suddenly after a few iterations.

**Explanation:**
A high learning rate can cause parameter updates to overshoot local minima, pushing activations and gradients into unstable numerical regions (e.g., sigmoid saturation, `inf` values in softmax).

**Fixes:**
- Use a learning rate finder or a scheduler (cosine decay, OneCycle).
- Apply gradient clipping (``torch.nn.utils.clip_grad_norm_``).
- Use adaptive optimizers (AdamW, Adafactor) with warm-up.

Weak Pretraining of VAE Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Context:**
In models like PairVAE, each VAE should first learn a stable latent representation. If the VAEs are undertrained or poorly converged, their latent spaces remain unstructured.

**Result:**
The joint or contrastive loss (e.g., Barlow Twins loss) operates on meaningless or high-variance latent representations, which can lead to exploding gradients or NaN losses.

**Fixes:**
- Ensure each VAE is pretrained until its reconstruction and KL losses stabilize.
- Visualize latent distributions to confirm approximate normality.
- Gradually unfreeze or jointly fine-tune after pretraining.

Non-normalized Data
^^^^^^^^^^^^^^^^^^^
**Symptoms:**
- High variance in activations.
- Gradients blowing up early in training.

**Explanation:**
Input data with large or inconsistent scales amplifies activations and gradients, destabilizing the forward and backward passes.

**Fixes:**
- Normalize inputs with help of the documentation.
- Check for outliers in the dataset.


