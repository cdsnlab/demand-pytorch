import math 
import time
import numpy as np 
import pandas as pd
from datetime import datetime
from data.utils import *
from data.datasets import STSSLDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min



class STSSLTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger(self.save_name)
    
    def normalize_data(self, data, scalar_type='Standard'):
        scalar = None
        if scalar_type == 'MinMax01':
            scalar = MinMax01Scaler(min=data.min(), max=data.max())
        elif scalar_type == 'MinMax11':
            scalar = MinMax11Scaler(min=data.min(), max=data.max())
        elif scalar_type == 'Standard':
            scalar = StandardScaler(mean=data.mean(), std=data.std())
        else:
            raise ValueError('scalar_type is not supported in data_normalization.')
        # print('{} scalar is used!!!'.format(scalar_type))
        # time.sleep(3)
        return scalar

    def get_dataset(self, data_dir, dataset, scalar_type='Standard'):
        datasets = {}
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'))
            datasets[category] = {'x': cat_data['x'], 'y': cat_data['y']}
        self.scaler = self.normalize_data(np.concatenate([datasets['train']['x'], datasets['val']['x']], axis=0), scalar_type)
        
        # Data format
        for category in ['train', 'val', 'test']:
            datasets[category]['x'] = self.scaler.transform(datasets[category]['x'])
            datasets[category]['y'] = self.scaler.transform(datasets[category]['y'])

        return datasets

    def load_adj(self, data_dir, dataset):
        self.adj = np.load(os.path.join(data_dir, dataset, 'adj_mx.npz'))['adj_mx']
        self.adj = torch.tensor(self.adj, device=self.device, dtype=torch.float)

    def load_dataset(self):
        self.load_adj(self.config.dataset_dir, self.config.dataset_name)
        datasets = self.get_dataset(self.config.dataset_dir, self.config.dataset_name)
        return datasets

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = STSSLDataset(datasets[category])
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(len(datasets['val']) / self.config.test_batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        return datasets['train'], datasets['val'], datasets['test']
    
    def compose_loader(self):
        train_dataset, val_dataset, test_dataset = self.compose_dataset()
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Test [{}] Val [{}]'.format(toGreen(len(train_dataset)), toGreen(len(test_dataset)), toGreen(len(val_dataset))))
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.test_batch_size, shuffle=False)
    

    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).float()
            target = target.to(self.device).float()

            self.optimizer.zero_grad()

            repr1, repr2 = self.model(data, self.adj)

            train_loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler)
            train_loss.backward()

            self.optimizer.step()

            training_time = time.time() - start_time
            start_time = time.time()

            with torch.no_grad():
                repr1, repr2 = self.model(data, self.adj)
                output = self.model.predict(repr1, repr2)

            # rescale
            output = self.scaler.inverse_transform(output)
            target = self.scaler.inverse_transform(target)

            loss = self.loss(output, target) 
            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def validate_epoch(self, epoch, is_test):
        self.model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.test_loader if is_test else self.val_loader):
            data = data.to(self.device).float()
            target = target.to(self.device).float()

            with torch.no_grad():
                repr1, repr2 = self.model(data, self.adj)
                output = self.model.predict(repr1, repr2)

            #rescale
            output = self.scaler.inverse_transform(output)
            target = self.scaler.inverse_transform(target)

            loss = self.loss(output, target) 

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics