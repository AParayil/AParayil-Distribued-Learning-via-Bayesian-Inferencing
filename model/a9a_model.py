import torch
import torch.nn as nn
import torch.distributions.laplace

class a9a_model(nn.Module):
    def __init__(self, num_inputs):
        super(a9a_model, self).__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self._initialize_parameters()

    def _initialize_parameters(self):
        m = torch.distributions.laplace.Laplace(torch.Tensor([0.0]), torch.Tensor([1.0]))
        w = m.rsample(self.linear.weight.shape).reshape(self.linear.weight.shape[0], self.linear.weight.shape[1])
        b = m.rsample(self.linear.bias.shape).reshape(self.linear.bias.shape[0])
        self.linear.weight = nn.Parameter(w)
        self.linear.bias = nn.Parameter(b)

    def forward(self, x):
        out = self.linear(x)
        return out
