import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl

class WarmupReduceLROnPlateau(torch.optim.lr_scheduler._LRScheduler):
    """
    Scheduler combinant :
    - Warmup linéaire pendant `warmup_epochs`
    - Puis ReduceLROnPlateau ensuite
    """

    def __init__(self, optimizer, warmup_epochs, max_lr, eta_min, reduce_on_plateau_args):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.eta_min = eta_min

        # Plateau scheduler intégré
        self.plateau = ReduceLROnPlateau(optimizer, **reduce_on_plateau_args)
        self.finished_warmup = False
        self.last_epoch = -1

        super().__init__(optimizer)

    def get_lr(self):
        # Si warmup encore actif
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / float(self.warmup_epochs)
            return [
                self.eta_min + scale * (self.max_lr - self.eta_min)
                for _ in self.base_lrs
            ]
        else:
            # Après warmup → garder le LR actuel, plateau gère la suite
            if not self.finished_warmup:
                self.finished_warmup = True
            return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics=None, epoch=None):
        self.last_epoch += 1
        if self.last_epoch < self.warmup_epochs:
            # Phase de warmup
            lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        else:
            # Phase ReduceLROnPlateau (après warmup)
            if metrics is not None:
                self.plateau.step(metrics)