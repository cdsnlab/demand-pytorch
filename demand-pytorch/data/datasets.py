import torch 
import numpy as np
from torch.utils.data import Dataset


class ConvLSTMDataset(Dataset):
    def __init__(self, data):
        x = data['x'].reshape(data['x'].shape[0], data['x'].shape[1], 32*32, 2)
        x = x.reshape(x.shape[0], x.shape[1], 32, 32, 2)
        self.x = x.transpose((0, 1, 4, 2, 3))

        y = data['y'].reshape(data['y'].shape[0], data['y'].shape[1], 32*32, 2)
        y = y.reshape(y.shape[0], y.shape[1], 32, 32, 2)
        self.y = y.transpose((0, 1, 4, 2, 3))
    
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]


class STMGCNDataset(Dataset):
    def __init__(self, data):
        self.x = data['x'].reshape(data['x'].shape[0], data['x'].shape[1], 32*32, 2)
        self.y = data['y'].reshape(data['y'].shape[0], data['y'].shape[1], 32*32, 2)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]
    
    
class DeepSTNDataset(Dataset):
    def __init__(self, data):
        self.x = data['x'].reshape(data['x'].shape[0], data['x'].shape[1], 32*32, 2)
        self.y = data['y'].reshape(data['y'].shape[0], data['y'].shape[1], 32*32, 2)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]
