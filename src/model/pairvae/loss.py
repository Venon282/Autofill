"""Loss functions used by the PairVAE model."""

import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    """Implementation of the Barlow Twins loss for latent alignment."""

    def __init__(self, lambda_coeff: float = 5e-3) -> None:
        super().__init__()
        self.lambda_coeff = lambda_coeff

    @staticmethod
    def off_diagonal_elements(matrix: torch.Tensor) -> torch.Tensor:
        """Return a flattened view of the off-diagonal elements of ``matrix``."""

        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError("Input must be a square matrix")
        return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()

    def forward(self, z_saxs: torch.Tensor, z_les: torch.Tensor) -> torch.Tensor:
        """Compute the correlation alignment loss between two latent batches."""

        if torch.isnan(z_saxs).any() or torch.isinf(z_saxs).any():
            raise RuntimeError("BarlowTwinsLoss: z_saxs contains NaN or inf values")
        if torch.isnan(z_les).any() or torch.isinf(z_les).any():
            raise RuntimeError("BarlowTwinsLoss: z_les contains NaN or inf values")

        batch_size = z_saxs.shape[0]
        std_saxs = torch.std(z_saxs, dim=0)
        std_les = torch.std(z_les, dim=0)
        if (std_saxs == 0).any():
            raise RuntimeError("BarlowTwinsLoss: std(z_saxs) is zero for some dimensions")
        if (std_les == 0).any():
            raise RuntimeError("BarlowTwinsLoss: std(z_les) is zero for some dimensions")

        safe_std_saxs = std_saxs.clone()
        safe_std_saxs[safe_std_saxs == 0] = 1.0
        safe_std_les = std_les.clone()
        safe_std_les[safe_std_les == 0] = 1.0

        z_saxs_norm = (z_saxs - torch.mean(z_saxs, dim=0)) / safe_std_saxs
        z_les_norm = (z_les - torch.mean(z_les, dim=0)) / safe_std_les
        if torch.isnan(z_saxs_norm).any() or torch.isinf(z_saxs_norm).any():
            raise RuntimeError("BarlowTwinsLoss: normalised z_saxs contains NaN or inf values")
        if torch.isnan(z_les_norm).any() or torch.isinf(z_les_norm).any():
            raise RuntimeError("BarlowTwinsLoss: normalised z_les contains NaN or inf values")

        cross_corr = torch.matmul(z_saxs_norm.T, z_les_norm) / batch_size
        if torch.isnan(cross_corr).any() or torch.isinf(cross_corr).any():
            raise RuntimeError("BarlowTwinsLoss: cross-correlation contains NaN or inf values")

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_elements(cross_corr).pow_(2).sum()
        return on_diag + self.lambda_coeff * off_diag
