from src.model.vae.submodel.ResVAE import ResVAE
from src.model.vae.submodel.VAE import VAE

MODEL_REGISTRY = {
    "VAE": VAE,
    "ResVAE": ResVAE,
}
