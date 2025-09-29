import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

import math 

from src.model.pairvae.loss import BarlowTwinsLoss
from src.model.pairvae.pairvae import PairVAE


class PlPairVAE(pl.LightningModule):
    """
    Classe PairVAE pour l'entraînement par paire avec reconstruction croisée
    et alignement des espaces latents.
    """

    def __init__(self, config, load_weights_VAE=True):
        super(PlPairVAE, self).__init__()

        self.config = config
        self.model = PairVAE(self.config["model"])

        self.barlow_twins_loss = BarlowTwinsLoss(self.config["training"]["lambda_param"])

        self.weight_latent_similarity = self.config["training"]["weight_latent_similarity"]
        self.weight_saxs2saxs = self.config["training"]["weight_saxs2saxs"]
        self.weight_saxs2les = self.config["training"]["weight_saxs2les"]
        self.weight_les2les = self.config["training"]["weight_les2les"]
        self.weight_les2saxs = self.config["training"]["weight_les2saxs"]

        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)

    def compute_loss(self, batch, outputs):
        """
        Paramètres:
            batch (dict): Contient "data_y_saxs" (SAXS) et "data_y_les" (LES).

        Renvoie:
            tuple: (loss_total, details) où details est un dictionnaire des pertes individuelles.
        """

        loss_saxs2saxs = F.mse_loss(outputs["recon_saxs"], batch["data_y_saxs"])
        loss_les2les = F.mse_loss(outputs["recon_les"], batch["data_y_les"])
        loss_saxs2les = F.mse_loss(outputs["recon_saxs2les"], batch["data_y_les"])
        loss_les2saxs = F.mse_loss(outputs["recon_les2saxs"], batch["data_y_saxs"])

        loss_latent = self.barlow_twins_loss(outputs["z_saxs"], outputs["z_les"])

        details = {
            "loss_latent": loss_latent.item(),
            "loss_saxs2saxs": loss_saxs2saxs.item(),
            "loss_les2les": loss_les2les.item(),
            "loss_saxs2les": loss_saxs2les.item(),
            "loss_les2saxs": loss_les2saxs.item()
        }

        loss_total = (self.weight_latent_similarity * loss_latent +
                      self.weight_saxs2saxs * loss_saxs2saxs +
                      self.weight_les2les * loss_les2les +
                      self.weight_saxs2les * loss_saxs2les +
                      self.weight_les2saxs * loss_les2saxs)

        return loss_total, details

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        loss_total, details = self.compute_loss(batch, outputs)

        self.log('train_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_latent', details["loss_latent"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)
        self.log('train_loss_saxs2saxs', details["loss_saxs2saxs"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)
        self.log('train_loss_les2les', details["loss_les2les"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)
        self.log('train_loss_saxs2les', details["loss_saxs2les"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)
        self.log('train_loss_les2saxs', details["loss_les2saxs"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)

        return loss_total

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        loss_total, details = self.compute_loss(batch, outputs)

        self.log('val_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss_latent', details["loss_latent"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_loss_saxs2saxs', details["loss_saxs2saxs"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)
        self.log('val_loss_les2les', details["loss_les2les"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)
        self.log('val_loss_saxs2les', details["loss_saxs2les"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)
        self.log('val_loss_les2saxs', details["loss_les2saxs"], on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["max_lr"]
        )

        # --- Warm-up settings ---
        warmup_epochs = self.config["training"].get("warmup_epochs", 5)
        max_epochs = self.config["training"]["num_epochs"]

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Phase de warm-up linéaire
                return float(current_epoch + 1) / float(warmup_epochs)
            else:
                # Phase de scheduler classique (cosine ici)
                progress = (current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine annealing

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def les_to_saxs(self, batch):
        
        output_les = self.model.vae_les(batch)
        recon_les2saxs = self.model.vae_saxs.decode(output_les["z"])

        return recon_les2saxs, batch["data_q"]

    def saxs_to_les(self, batch):
        
        output_saxs = self.model.vae_saxs(batch)
        recon_saxs2les = self.model.vae_les.decode(output_saxs["z"])

        return recon_saxs2les, batch["data_q"]

    def saxs_to_saxs(self, batch):

        output_saxs = self.model.vae_saxs(batch)
        recon_saxs2saxs = self.model.vae_saxs.decode(output_saxs["z"])

        return recon_saxs2saxs, batch["data_q"]

    def les_to_les(self, batch):

        output_les = self.model.vae_les(batch)
        recon_les2les = self.model.vae_les.decode(output_les["z"])

        return recon_les2les, batch["data_q"]

    def get_transforms_data_les(self):
        config = self.model.get_les_config()
        return config["transforms_data"]

    def get_transforms_data_saxs(self):
        config = self.model.get_saxs_config()
        return config["transforms_data"]
