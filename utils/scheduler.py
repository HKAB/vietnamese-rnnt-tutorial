import torch

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [(min(1.0, self._step_count / self.warmup_steps)) * base_lr for base_lr in self.base_lrs]
        else:
            # Exponential decay phase
            decay_factor = (self.min_lr / self.base_lrs[0]) ** ((self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            return [decay_factor * base_lr for base_lr in self.base_lrs]