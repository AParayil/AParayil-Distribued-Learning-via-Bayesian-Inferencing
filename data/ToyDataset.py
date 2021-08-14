from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    """Mixture of Gaussian toy dataset."""

    def __init__(self, num_samples, theta1, theta2, sigmax):
        """
        Args:
            num_samples: Number of data samples
        """
        self.data = torch.zeros(num_samples, 1)
        for i in range(num_samples):
            if np.random.rand(1) > 0.5:
                self.data[i, :] = torch.Tensor([theta1]) + torch.randn(1, 1) * sigmax
            else:
                self.data[i, :] = torch.Tensor([theta1 + theta2]) + torch.randn(1, 1) * sigmax

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index, :]

        return x
