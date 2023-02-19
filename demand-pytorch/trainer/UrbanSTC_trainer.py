import math 
import time
import numpy as np 
from data.utils import *
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger
from model import UrbanSTC

class UrbanSTCTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger(self.save_name)
    
    def load_dataset(self):
        Tensor = torch.cuda.FloatTensor
        
        X_train = np.load(self.config.dataset_dir + "/P1/train/" + "X.npy")
        X_valid = np.load(self.config.dataset_dir + "/P1/valid/" + "X.npy")
        X_test = np.load(self.config.dataset_dir + "/P1/test/" + "X.npy")
        Y_train = np.load(self.config.dataset_dir + "/P1/train/" + "Y.npy")
        Y_valid = np.load(self.config.dataset_dir + "/P1/valid/" + "Y.npy")
        Y_test = np.load(self.config.dataset_dir + "/P1/test/" + "Y.npy")
        ext_train = np.load(self.config.dataset_dir + "/P1/train/" + "ext.npy")
        ext_valid = np.load(self.config.dataset_dir + "/P1/valid/" + "ext.npy")
        ext_test = np.load(self.config.dataset_dir + "/P1/test/" + "ext.npy")
        

        X_train = Tensor(np.expand_dims(X_train, 1))
        Y_train = Tensor(np.expand_dims(Y_train, 1))
        ext_train = Tensor(ext_train)
        X_valid = Tensor(np.expand_dims(X_valid, 1))
        Y_valid = Tensor(np.expand_dims(Y_valid, 1))
        ext_valid = Tensor(ext_valid)
        X_test = Tensor(np.expand_dims(X_test, 1))
        Y_test = Tensor(np.expand_dims(Y_test, 1))
        ext_test = Tensor(ext_test)

        
        data_train = torch.utils.data.TensorDataset(X_train, ext_train, Y_train)
        data_valid = torch.utils.data.TensorDataset(X_valid, ext_valid, Y_valid)
        data_test = torch.utils.data.TensorDataset(X_test, ext_test, Y_test)
       
        datasets = {'train': data_train, 'val': data_valid, 'test':data_test}
        
        return datasets

    def setup_model(self):
        self.model = UrbanSTC.UrbanSTCModel()
        self.model.cuda()
        self.model.apply(UrbanSTC.weights_init_normal)
        torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=5.0)
        self.model.conv1.load_state_dict(torch.load('../results/inf_pretrain_model'))
        self.model.conv_tc.load_state_dict(torch.load('../results/tc_pretrain_model'))
        self.model.conv_pix.load_state_dict(torch.load('../results/reg_pretrain_model'))
        for p in self.model.conv1.parameters():
            p.requires_grad = False
        for p in self.model.conv_tc.parameters():
            p.requires_grad = False
        for p in self.model.conv_pix.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()))

    def compose_dataset(self):
        datasets = self.load_dataset()
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(len(datasets['val']) / self.config.test_batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        return datasets
    
    def compose_loader(self):
        datasets = self.compose_dataset()
        
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        test_dataset = datasets['test']
        
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Test [{}] Val [{}]'.format(toGreen(len(train_dataset)), toGreen(len(test_dataset)), toGreen(len(val_dataset))))
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.test_batch_size, shuffle=False)


    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, ext, target) in enumerate(self.train_loader):
            data,ext, target = data.to(self.device).float(),ext.to(self.device).float(), target.to(self.device).float()
            self.optimizer.zero_grad()

            output = self.model(data, ext)
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

        for batch_idx, (data, ext, target) in enumerate(self.test_loader if is_test else self.val_loader):
            data, target = data.to(self.device).float(), target.to(self.device).float()

            with torch.no_grad():
                output = self.model(data, ext)
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

