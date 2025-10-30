from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class VAEModelConfig(BaseModel):
    """Model configuration for a single-domain VAE."""
    vae_class: str = Field(..., description="Registered VAE architecture name in MODEL_REGISTRY.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the submodel constructor.")
    beta: float = Field(default=1e-6, ge=0.0, description="KL divergence scaling coefficient.")
    data_q: Optional[Any] = None
    transforms_data: Optional[Dict[str, Any]] = None


class VAETrainingConfig(BaseModel):
    """Training configuration for the VAE Lightning module."""
    max_lr: float = Field(default=1e-3, gt=0)
    warmup_epochs: int = Field(default=5, ge=0)
    num_epochs: int = Field(default=100, ge=1)
    weighted_loss: bool = False
    weighted_loss_limit_index: Optional[int] = None
    eta_min: float = 1e-15