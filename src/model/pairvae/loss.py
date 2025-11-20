"""Loss functions used by the PairVAE model."""

import torch
import torch.nn as nn

from src.logging_utils import get_logger
logger = get_logger(__name__)


class BarlowTwinsLoss(nn.Module):
    """
    The Barlow-Twinn Loss normalizes the D dimensinonal vectors from the projection head and then computes the DxD cross-correlation matrix between the normalized vectors of the 2 views of each signal.

    Then it splits this cross-correlation matrix into two parts. The first part, the diagonal of this matrix is brought closer to 1, 
    which pushes up the cosine similarity between the latent vectors of two views of each signal, thus making the backbone invariant to the transformations applied to the views. 
    The second part of the loss pushes the non-diagonal elements of the cross-corrlelation matrix closes to 0.
    This reduces the redundancy between the different dimensions of the latent vector.

    """
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()

        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        eps = 1e-8
        z1_std = torch.std(z1, dim=0, correction=0)
        z2_std = torch.std(z2, dim=0, correction=0)

        z1_norm = (z1 - torch.mean(z1, dim=0)) / (z1_std + eps)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / (z2_std + eps)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / z1.size(0)

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag

