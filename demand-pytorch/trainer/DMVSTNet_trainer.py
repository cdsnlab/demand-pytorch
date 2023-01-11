import math 
import time
import numpy as np 
from data.utils import *
from data.datasets import DMVSTNetDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger

class DMVSTNetTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger(self.save_name)
    
    def load_dataset(self):
        with open(os.path.join(self.config.dataset_dir, self.config.dataset_name+".pickle"), "rb") as file:
            data = pickle.load(file)
        datasets = {}
        for category in ['train', 'test']:
            datasets[category] = data[category]
        return datasets

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'test']:
            datasets[category] = DMVSTNetDataset(datasets[category])
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        return datasets['train'], datasets['test']
    
    def compose_loader(self):
        train_dataset, test_dataset = self.compose_dataset()
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Test [{}]'.format(toGreen(len(train_dataset)), toGreen(len(test_dataset))))
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.val_loader = self.test_loader

    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (cnn, flow, topo, target) in enumerate(self.train_loader):
            cnn, flow, topo, target = cnn.to(self.device).float(), flow.to(self.device).float(), topo.to(self.device).float(), target.to(self.device).float()
            self.optimizer.zero_grad()

            output = self.model(cnn, flow, topo)
            output = output[:, -self.config.num_pred:]

            output *= self.config.train_norm
            target *= self.config.train_norm

            loss = self.loss(output, target) 
            loss.backward()
            self.optimizer.step()

            training_time = time.time() - start_time
            start_time = time.time()

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

        for batch_idx, (cnn, flow, topo, target) in enumerate(self.test_loader):
            cnn, flow, topo, target = cnn.to(self.device).float(), flow.to(self.device).float(), topo.to(self.device).float(), target.to(self.device).float()

            with torch.no_grad():
                output = self.model(cnn, flow, topo)
            output = output[:, -self.config.num_pred:]

            output *= self.config.test_norm
            target *= self.config.test_norm

            loss = self.loss(output, target) 

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, batch_idx, \
                self.num_test_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics

