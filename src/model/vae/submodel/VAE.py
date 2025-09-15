import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, in_channels=1,
                 down_channels=[32, 64, 128], up_channels=[128, 64, 32], dilation=1,
                 output_channels=1, strat="y", *args, **kwargs):
        super(VAE, self).__init__()

        if len(down_channels) != len(up_channels):
            raise ValueError("down_channels et up_channels doivent avoir la même taille.")
        if down_channels[-1] != up_channels[0]:
            raise ValueError("Le dernier canal de down_channels doit être égal au premier canal de up_channels.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.strat = strat

        encoder_layers = []
        current_in = in_channels
        for out_ch in down_channels:
            encoder_layers.append(nn.Conv1d(current_in, out_ch, kernel_size=3, stride=2, padding=1, dilation=dilation))
            # encoder_layers.append(nn.BatchNorm1d(out_ch))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1))
            encoder_layers.append(nn.ReLU())
            # encoder_layers.append(nn.MaxPool1d(kernel_size=2))
            current_in = out_ch
            input_dim = input_dim // 2 if input_dim % 2 == 0 else input_dim // 2 + 1
        encoder_layers.append(nn.Flatten())
        flattened_size = down_channels[-1] * input_dim
        encoder_layers.append(nn.Linear(flattened_size, flattened_size // 2))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(flattened_size // 2, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size // 2, latent_dim)

        decoder_layers = []
        current_in = up_channels[0]
        for out_ch in up_channels[1:]:
            decoder_layers.append(
                nn.ConvTranspose1d(current_in, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            # decoder_layers.append(nn.BatchNorm1d(out_ch))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1))  # Convolution supplémentaire
            decoder_layers.append(nn.ReLU())
            current_in = out_ch
        decoder_layers.append(
            nn.ConvTranspose1d(current_in, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, up_channels[0] * input_dim),
            nn.ReLU(),
            nn.Unflatten(1, (up_channels[0], input_dim)),
            *decoder_layers,
            nn.Upsample(size=self.input_dim, mode='linear', align_corners=True)
        )
        self.display_info()

    def display_info(self):
        test_tensor = torch.zeros(1, self.in_channels, self.input_dim)
        flattened_size = self.encoder(test_tensor).view(1, -1).size(1)
        print("VAE Architecture:")
        print("\tInput Dimension:", self.input_dim)
        print("\tLatent Dimension:", self.latent_dim)
        print("\tIn Channels:", self.in_channels)
        print("\tDown Channels:", self.down_channels)
        print("\tUp Channels:", self.up_channels)
        print("\tOutput Channels:", self.output_channels)
        print("\tFlattened Size:", flattened_size)
        print("\tEncoder Architecture:", self.encoder)
        print("\tDecoder Architecture:", self.decoder)

    def encode(self, x):
        """ Encodeur VAE """
        for layer in self.encoder:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """ Reparamétrisation de l'espace latent """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """ Décodeur VAE """
        assert not torch.isnan(z).any(), "NaN detected in input latent variable z"

        for i, layer in enumerate(self.decoder):
            z = layer(z)
            assert not torch.isnan(z).any(), f"NaN detected after layer {i}"
        return z

    def forward(self, y, q=None, metadata=None):
        if self.strat == "y":
            x = y
        else:
            raise ValueError("strat must be 'q' or 'y'")

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError("[VAE] NaN or inf detected in input x")

        mu, logvar = self.encode(x)
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            raise RuntimeError("[VAE] NaN or inf detected in mu")
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            raise RuntimeError("[VAE] NaN or inf detected in logvar")

        z = self.reparameterize(mu, logvar)
        if torch.isnan(z).any() or torch.isinf(z).any():
            raise RuntimeError("[VAE] NaN or inf detected in latent variable z")

        recon = self.decode(z)
        if torch.isnan(recon).any() or torch.isinf(recon).any():
            raise RuntimeError("[VAE] NaN or inf detected in reconstructed output")

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z
        }
