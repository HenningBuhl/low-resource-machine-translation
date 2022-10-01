from torch.optim.lr_scheduler import _LRScheduler


import torch


class WarumUpInverseSquareRootScheduler(_LRScheduler):
    '''A class implementing warm-up into inverse square root scheduling.'''

    def __init__(self, optimizer, d_model, warm_up_steps, factor=1, verbose=False):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warm_up_steps = warm_up_steps
        self.factor = factor
        super(WarumUpInverseSquareRootScheduler, self).__init__(optimizer, verbose=verbose)  # TODO causes TypeError: super(type, obj): obj must be an instance or subtype of type (must restart kernel).

    def get_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.factor * self.d_model**(-0.5) * min(self._step_count **(-0.5), self._step_count * self.warm_up_steps**(-1.5))
        return [group['lr'] for group in self.optimizer.param_groups]
