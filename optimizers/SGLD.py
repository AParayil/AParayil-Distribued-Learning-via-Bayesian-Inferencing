#### Modified from https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics/optimizers.py

from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch

class sgld(Optimizer):

    def __init__(self, params, lr=required, norm_sigma=0, num_batches=required, addnoise=True):

        if norm_sigma == 0:
            weight_decay = 0.0
        else:
            weight_decay = 1 / (norm_sigma ** 2)

        defaults = dict(lr=lr, weight_decay=weight_decay, batch_weight=num_batches, addnoise=addnoise)

        super(sgld, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            batch_weight = group['batch_weight']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = batch_weight * p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if group['addnoise']:

                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                    p.data.add_(-group['lr'], 0.5 * d_p + langevin_noise)

                else:
                    p.data.add_(-group['lr'], 0.5 * d_p)

        return loss
