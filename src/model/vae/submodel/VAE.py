import torch
from torch import nn

from src.logging_utils import get_logger


logger = get_logger(__name__)


class VAE(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=64, in_channels=1, hidden_dims=None, output_channels=1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]  # 5 couches

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # --- ENCODER ---
        modules = []
        current_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(current_in, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU()
                )
            )
            current_in = h_dim
        self.encoder = nn.Sequential(*modules)

        # Calcul de la taille compressée automatiquement
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_dim)
            encoded = self.encoder(dummy)
            self.compressed_length = encoded.shape[-1]

        encoder_output_dim = hidden_dims[-1] * self.compressed_length
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)

        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, encoder_output_dim)

        hidden_dims_rev = hidden_dims[::-1]

        modules = []
        for i in range(len(hidden_dims_rev) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims_rev[i],
                        hidden_dims_rev[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm1d(hidden_dims_rev[i + 1]),
                    nn.GELU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        # --- Dernière couche de sortie ---
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dims_rev[-1],
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1) )

    def display_info(self):
        logger.info("RESEAU VAE")

    # ---- ENCODER ----
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    # ---- LATENT SAMPLING ----
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- DECODER ----
    def decode(self, z):
        recon = self.decoder_input(z)
        recon = recon.view(-1, self.encoder[-1][0].out_channels, self.compressed_length)
        recon = self.decoder(recon)
        recon = self.final_layer(recon)

        # Ajustement automatique de taille si nécessaire
        if recon.size(-1) > self.input_dim:
            recon = recon[..., :self.input_dim]
        elif recon.size(-1) < self.input_dim:
            diff = self.input_dim - recon.size(-1)
            recon = torch.nn.functional.pad(recon, (0, diff))

        return recon

    # ---- FORWARD ----
    def forward(self, x, metadata=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
