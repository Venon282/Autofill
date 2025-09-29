import torch
import torch.nn as nn


# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/barlow-twins.html#Barlow-Twins-Loss

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_saxs, zles):
        batch_size, feature_dim = z_saxs.shape

        # N x D, where N is the batch size and D is output dim of projection head
        z_saxs_norm = (z_saxs - torch.mean(z_saxs, dim=0)) / torch.std(z_saxs, dim=0)
        zles_norm = (zles - torch.mean(zles, dim=0)) / torch.std(zles, dim=0)

        # Vérification NaN/infs dans les entrées AVANT normalisation

        if torch.isnan(z_saxs).any() or torch.isinf(z_saxs).any():
            raise RuntimeError("[BarlowTwinsLoss] z_saxs (input) contient NaN ou inf!")
        if torch.isnan(zles).any() or torch.isinf(zles).any():
            raise RuntimeError("[BarlowTwinsLoss] zles (input) contient NaN ou inf!")

        # Vérification des std
        std_saxs = torch.std(z_saxs, dim=0)
        std_les = torch.std(zles, dim=0)
        if (std_saxs == 0).any():
            raise RuntimeError(f"[BarlowTwinsLoss] std(z_saxs) == 0 pour indices : {torch.where(std_saxs == 0)[0].cpu().numpy()}")
        if (std_les == 0).any():
            raise RuntimeError(f"[BarlowTwinsLoss] std(zles) == 0 pour indices : {torch.where(std_les == 0)[0].cpu().numpy()}")

        # Normalisation robuste : évite division par zéro
        safe_std_saxs = std_saxs.clone()
        safe_std_saxs[safe_std_saxs == 0] = 1.0
        safe_std_les = std_les.clone()
        safe_std_les[safe_std_les == 0] = 1.0
        
        z_saxs_norm = (z_saxs - torch.mean(z_saxs, dim=0)) / safe_std_saxs
        zles_norm = (zles - torch.mean(zles, dim=0)) / safe_std_les


        if torch.isnan(z_saxs_norm).any() or torch.isinf(z_saxs_norm).any():
            raise RuntimeError("[BarlowTwinsLoss] z_saxs_norm contient NaN ou inf!")
        if torch.isnan(zles_norm).any() or torch.isinf(zles_norm).any():
            raise RuntimeError("[BarlowTwinsLoss] zles_norm contient NaN ou inf!")

        cross_corr = torch.matmul(z_saxs_norm.T, zles_norm) / batch_size


        if torch.isnan(cross_corr).any() or torch.isinf(cross_corr).any():
            raise RuntimeError("[BarlowTwinsLoss] cross_corr contient NaN ou inf!")

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        loss = on_diag + self.lambda_coeff * off_diag
        return loss
