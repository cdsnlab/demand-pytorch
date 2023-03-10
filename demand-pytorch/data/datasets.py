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

class STResNetDataset(Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']

    def __getitem__(self, index):
        x = (torch.tensor(self.x[0][index]), torch.tensor(self.x[1][index]), torch.tensor(self.x[2][index]), torch.tensor(self.x[3][index]))
        y = self.y[index]
        return x, torch.tensor(y)

    def __len__(self):
        return self.x[0].shape[0]

class STSSLDataset(Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]

class DMVSTNetDataset(Dataset):
    def __init__(self, data):
        self.cnn = data['cnn']
        self.flow = data['flow']
        self.topo = data['topo']
        self.y = data['y']

    def __getitem__(self, index):
        cnn = torch.tensor(self.cnn[index])
        flow = torch.tensor(self.flow[index]).permute(dims=(0, 3, 1, 2))
        topo = torch.tensor(self.topo[index]).permute(dims=(2, 0, 1)).reshape(-1, 10, 20)
        y = torch.tensor(self.y[index])
        return cnn, flow, topo, y

    def __len__(self):
        return self.cnn.shape[0]

class MDLDataset(Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']
        print(self.x[0].shape)
        print(self.x[1].shape)
        print(self.x[2].shape)
        print(self.x[3].shape)
        print(self.x[4].shape)
        print(self.x[5].shape)
        print(self.x[6].shape)
        print()

    def __getitem__(self, index):
        x = (torch.tensor(self.x[0][index]), torch.tensor(self.x[1][index]), torch.tensor(self.x[2][index]), torch.tensor(self.x[3][index]), torch.tensor(self.x[4][index]), torch.tensor(self.x[5][index]), torch.tensor(self.x[6][index])) 
        y = (torch.tensor(self.y[0][index]), torch.tensor(self.y[1][index]))
        return x, y

    def __len__(self):
        return self.x[0].shape[0]
class DeepSTNDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]
    
class ST_MetaNetDataset(Dataset):
    def __init__(self, data):
        self.x = data['x'].reshape(data['x'].shape[0], data['x'].shape[1], 32*32, 2)
        self.y = data['y'].reshape(data['y'].shape[0], data['y'].shape[1], 32*32, 2)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]
    
class STG2SeqDataset(Dataset):
    def __init__(self, data):
        self.x = data['x'].reshape(data['x'].shape[0], data['x'].shape[1], 32*32, 2)
        self.y = data['y'].reshape(data['y'].shape[0], data['y'].shape[1], 32*32, 2)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]
