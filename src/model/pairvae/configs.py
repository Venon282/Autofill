from pydantic import BaseModel, Field
from typing import Optional


class PairVAEModelConfig(BaseModel):
    type: str = Field(..., description="Type of model.")
    spec: str = Field("pair", description="Specification of the VAE model.")
    ckpt_path_saxs: Optional[str] = Field(None, description="Checkpoint path for pretrained SAXS VAE")
    ckpt_path_les: Optional[str] = Field(None, description="Checkpoint path for pretrained LES VAE")
    lr: float = Field(1e-4, ge=0)
    freeze_subvae: bool = False
    data_q_saxs: Optional[list[float]] = None
    data_q_les: Optional[list[float]] = None


class PairVAETrainingConfig(BaseModel):
    lambda_param: float = 0.005
    weight_latent_similarity: float = 1.0
    weight_saxs2saxs: float = 1.0
    weight_les2les: float = 1.0
    weight_saxs2les: float = 1.0
    weight_les2saxs: float = 1.0
    weighted_loss: bool = False
    weighted_loss_limit_index: Optional[int] = None
    max_lr: float = 1e-3
    num_epochs: int = 100
    warmup_epochs: int = 5    
    eta_min: float = 1e-15