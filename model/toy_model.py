import torch
import torch.nn as nn


class toy_model(nn.Module):
    def __init__(self, sigmax, sigma1, sigma2):
        super(toy_model, self).__init__()
        self.param1 = nn.Parameter(torch.randn(1) * sigma1)
        self.param2 = nn.Parameter(torch.randn(1) * sigma2)
        self.sigmax = torch.Tensor([sigmax])
        self.piconst = torch.Tensor([2 * 3.141592653])

    def forward(self, x, device):

        sigmax = self.sigmax.to(device)
        piconst = self.piconst.to(device)
        mixcomp = torch.Tensor([0.5]).to(device)
        a1 = - ((x - self.param1) ** 2) / (2.0 * (sigmax ** 2))
        a2 = - ((x - self.param1 - self.param2) ** 2) / (2 * (sigmax ** 2))
        ct = 1 / (torch.sqrt(piconst) * sigmax)

        log_p1 = torch.log(mixcomp) + torch.log(ct) + a1
        log_p2 = torch.log(mixcomp) + torch.log(ct) + a2

        log_p = self.logsumexp(log_p1, log_p2)

        return log_p

    def logsumexp(self, log_p1, log_p2):
        log_p_max = torch.max(log_p1, log_p2)
        res = log_p_max + torch.log((log_p1 - log_p_max).exp() + (log_p2 - log_p_max).exp())
        return res


