import torch
from torch import nn

from src.logging_utils import get_logger

logger = get_logger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.GELU = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # If input and output channels differ, use 1x1 convolution to match dimensions
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                         stride=2) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.GELU(out)
        out = self.conv2(out)
        out += identity  # Residual connection
        out = self.GELU(out)
        return out


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_padding=1):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1,
                                          output_padding=output_padding)
        self.GELU = nn.GELU()
        self.deconv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=1)

        self.skip_connection = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0,
                                                  output_padding=output_padding, bias=False)

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.deconv1(x)
        out = self.GELU(out)
        out = self.deconv2(out)
        out += identity  # Residual connection
        return out


class ResVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, in_channels=1,
                 down_channels=[16, 32, 64, 128, 256, 512], up_channels=[512, 256, 128, 64, 32, 16], dilation=1,
                 output_channels=1, strat="y", use_sigmoid=False, *args, **kwargs):
        super(ResVAE, self).__init__()

        if len(down_channels) != len(up_channels):
            raise ValueError("down_channels et up_channels doivent avoir la mÃªme taille.")
        if down_channels[-1] != up_channels[0]:
            raise ValueError("Le dernier canal de down_channels doit Ãªtre Ã©gal au premier canal de up_channels.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.strat = strat

        encoder_layers = []
        current_in = in_channels
        output_paddings=[]
        for out_ch in down_channels:
            output_paddings.insert(0, 1 if input_dim % 2 == 0 else  0)
            encoder_layers.append(ResidualBlock(current_in, out_ch))
            current_in = out_ch
            input_dim = input_dim // 2 if input_dim % 2 == 0 else input_dim // 2 + 1
        encoder_layers.append(nn.Flatten())

        flattened_size = down_channels[-1] * input_dim
        encoder_layers.append(nn.Linear(flattened_size, flattened_size // 2))
        encoder_layers.append(nn.GELU())
        encoder_layers.append(nn.Linear(flattened_size // 2, flattened_size // 4))
        encoder_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(flattened_size // 4, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size // 4, latent_dim)

        decoder_layers = []
        current_in = up_channels[0]
        for i, out_ch in enumerate(up_channels[1:]):
            decoder_layers.append(ResidualUpBlock(current_in, out_ch, kernel_size=3, output_padding=output_paddings[i]))
            decoder_layers.append(nn.GELU())
            current_in = out_ch
        decoder_layers.append(ResidualUpBlock(current_in, output_channels, kernel_size=3))

        layers = [
            nn.Linear(latent_dim, flattened_size // 4),
            nn.GELU(),
            nn.Linear(flattened_size // 4, flattened_size // 2),
            nn.GELU(),
            nn.Linear(flattened_size // 2, up_channels[0] * input_dim),
            nn.GELU(),
            nn.Unflatten(1, (up_channels[0], input_dim)),
            *decoder_layers,
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)
        # self.display_info()

    def display_info(self):
        test_tensor = torch.zeros(1, self.in_channels, self.input_dim)
        flattened_size = self.encoder(test_tensor).view(1, -1).size(1)
        logger.info("VAE Architecture:")
        logger.info("\tInput Dimension: %s", self.input_dim)
        logger.info("\tLatent Dimension: %s", self.latent_dim)
        logger.info("\tIn Channels: %s", self.in_channels)
        logger.info("\tDown Channels: %s", self.down_channels)
        logger.info("\tUp Channels: %s", self.up_channels)
        logger.info("\tOutput Channels: %s", self.output_channels)
        logger.info("\tFlattened Size: %s", flattened_size)
        logger.info("\tEncoder Architecture: %s", self.encoder)
        logger.info("\tDecoder Architecture: %s", self.decoder)

    # ---- ENCODER ----
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
        recon = self.decoder(z)
        return recon

    def forward(self, x, metadata=None):
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
