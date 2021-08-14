from __future__ import print_function, division
import torch
from torch.utils.data import Dataset


class a9aDataset(Dataset):

    def __init__(self, data, labels, train_idx, test_idx, train=True):

        self.train = train

        self.data = dict()
        self.data['traindata'] = torch.from_numpy(data[train_idx, :]).float()
        self.data['trainlabels'] = torch.squeeze(torch.from_numpy(labels[train_idx]).float())
        self.data['testdata'] = torch.from_numpy(data[test_idx, :]).float()
        self.data['testlabels'] = torch.squeeze(torch.from_numpy(labels[test_idx]).float())

    def __len__(self):

        if self.train:
            return self.data['traindata'].shape[0]
        else:
            return self.data['testdata'].shape[0]

    def __getitem__(self, index):

        if self.train:
            data, target = self.data['traindata'][index], self.data['trainlabels'][index]
        else:
            data, target = self.data['testdata'][index], self.data['testlabels'][index]

        return data, target
