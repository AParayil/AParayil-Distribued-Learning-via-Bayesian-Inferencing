#### Modified from https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics/optimizers.py

from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch

class dsgld(Optimizer):

    def __init__(self, params, allmodels=required, adj_vec=required, alpha=required, beta=required, norm_sigma1=required, norm_sigma2=required, num_batches=required, addnoise=True):

        weight_decay1 = 1 / (norm_sigma1 ** 2)
        weight_decay2 = 1 / (norm_sigma2 ** 2)

        defaults = dict(allmodels=allmodels, adj_vec=adj_vec, alpha=alpha, beta=beta, weight_decay1=weight_decay1, weight_decay2=weight_decay2, batch_weight=num_batches, addnoise=addnoise)

        super(dsgld, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            alpha = group['alpha']
            beta = group['beta']
            weight_decay1 = group['weight_decay1']
            weight_decay2 = group['weight_decay2']
            batch_weight = group['batch_weight']
            allmodels = group['allmodels']
            adj_vec = group['adj_vec']
            num_agents = len(allmodels)
            paramnum = 0
            stoch_grad = []

            for k in range(num_agents):

                if adj_vec[k] == 1:

                    n_params = allmodels[k].parameters()

                    for (p, neighp) in zip(group['params'], n_params):

                        p.data.add_(-beta, p.data - neighp.data)

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = batch_weight * p.grad.data

                if paramnum == 1 and weight_decay2 != 0:

                    d_p.add_(weight_decay2 / num_agents, p.data)

                if paramnum == 0 and weight_decay1 != 0:

                    d_p.add_(weight_decay1 / num_agents, p.data)

                    paramnum += 1

                if group['addnoise']:

                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(alpha / 2.0)

                    p.data.add_(-alpha, d_p + langevin_noise)

                else:

                    p.data.add_(-alpha, d_p)


        return loss