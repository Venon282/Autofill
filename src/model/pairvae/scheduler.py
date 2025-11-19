import torch
import lightning.pytorch as pl

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Scheduler combinant :
    - Warmup linéaire pendant `warmup_epochs`
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_lr: float,
        eta_min: float,
        step_size: int = 10,   # t=10 epochs
        gamma: float = 0.1,    # division par 10
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.eta_min = eta_min
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        # --- WARMUP ---
        if epoch <= self.warmup_epochs:
            scale = epoch / float(self.warmup_epochs)
            return [
                self.eta_min + scale * (self.max_lr - self.eta_min)
                for _ in self.base_lrs
            ]

        # --- STEP LR DECAY ---
        steps = (epoch - self.warmup_epochs) // self.step_size
        factor = self.gamma ** steps
        return [self.max_lr * factor for _ in self.base_lrs]


    def step(self, metrics=None, epoch=None):
        self.last_epoch += 1
        if self.last_epoch < self.warmup_epochs:
            # Phase de warmup
            lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        else:
            # Phase de descente
            lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr