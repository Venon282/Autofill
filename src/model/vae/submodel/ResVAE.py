import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.GELU = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip_connection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.GELU(out)
        out = self.conv2(out)
        out += identity
        out = self.GELU(out)
        return out


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Ajustement de output_padding=1 pour compenser les tailles impaires
        self.deconv1 = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            output_padding=1,  # clé pour rétablir la taille exacte
        )
        self.GELU = nn.GELU()
        self.deconv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=1)

        self.skip_connection = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            output_padding=1,  # idem ici
            bias=False,
        )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.deconv1(x)
        out = self.GELU(out)
        out = self.deconv2(out)
        out += identity
        out = self.GELU(out)
        return out


class ResVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        in_channels=1,
        down_channels=[32, 64, 128],
        up_channels=[128, 64, 32],
        output_channels=1,
        use_sigmoid=True,
    ):
        super().__init__()

        if len(down_channels) != len(up_channels):
            raise ValueError("down_channels et up_channels doivent avoir la même taille.")
        if down_channels[-1] != up_channels[0]:
            raise ValueError("Le dernier canal de down_channels doit être égal au premier de up_channels.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.output_channels = output_channels

        # --- ENCODER ---
        encoder_layers = []
        current_in = in_channels
        size_tracker = input_dim
        for out_ch in down_channels:
            encoder_layers.append(ResidualBlock(current_in, out_ch))
            current_in = out_ch
            size_tracker = (size_tracker + 1) // 2  # division par 2 à chaque bloc
        encoder_layers.append(nn.Flatten())

        flattened_size = down_channels[-1] * size_tracker
        encoder_layers += [
            nn.Linear(flattened_size, flattened_size // 2),
            nn.GELU(),
            nn.Linear(flattened_size // 2, flattened_size // 4),
            nn.GELU(),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(flattened_size // 4, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size // 4, latent_dim)

        # --- DECODER ---
        decoder_layers = []
        current_in = up_channels[0]
        for out_ch in up_channels[1:]:
            decoder_layers.append(ResidualUpBlock(current_in, out_ch, kernel_size=3))
            decoder_layers.append(nn.GELU())
            current_in = out_ch
        decoder_layers.append(ResidualUpBlock(current_in, output_channels, kernel_size=3))

        layers = [
            nn.Linear(latent_dim, flattened_size // 4),
            nn.GELU(),
            nn.Linear(flattened_size // 4, flattened_size // 2),
            nn.GELU(),
            nn.Linear(flattened_size // 2, up_channels[0] * size_tracker),
            nn.GELU(),
            nn.Unflatten(1, (up_channels[0], size_tracker)),
            *decoder_layers,
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x, metadata=None):
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # Ajustement final de la taille (au cas d’un pixel d’écart)
        if recon.size(-1) > x.size(-1):
            recon = recon[..., :x.size(-1)]
        elif recon.size(-1) < x.size(-1):
            diff = x.size(-1) - recon.size(-1)
            recon = torch.nn.functional.pad(recon, (0, diff))

        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}