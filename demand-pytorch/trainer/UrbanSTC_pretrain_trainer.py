import math 
import time
import numpy as np 
import importlib
import shutil
import json
from datetime import datetime
from data.utils import *
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger
from model.UrbanSTC import reg_preTrain, tc_preTrain, inference_net, weights_init_normal, loss_c, distance_tensor

class UrbanSTC_pretrainTrainer(BaseTrainer):
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
        
        anchor_train = np.load(self.config.dataset_dir + "/P1/train/" + "anchor.npy")
        anchor_valid = np.load(self.config.dataset_dir + "/P1/valid/" + "anchor.npy")
        anchor_test = np.load(self.config.dataset_dir + "/P1/test/" + "anchor.npy")
        pos_train = np.load(self.config.dataset_dir + "/P1/train/" + "pos.npy")
        pos_valid = np.load(self.config.dataset_dir + "/P1/valid/" + "pos.npy")
        pos_test = np.load(self.config.dataset_dir + "/P1/test/" + "pos.npy")
        neg_train = np.load(self.config.dataset_dir + "/P1/train/" + "neg.npy")
        neg_valid = np.load(self.config.dataset_dir + "/P1/valid/" + "neg.npy")
        neg_test = np.load(self.config.dataset_dir + "/P1/test/" + "neg.npy")
        
        X8_train = np.load(self.config.dataset_dir + "/P1/train/" + "8X.npy")
        X8_valid = np.load(self.config.dataset_dir + "/P1/valid/" + "8X.npy")
        X8_test = np.load(self.config.dataset_dir + "/P1/test/" + "8X.npy")
        
        X_train = Tensor(np.expand_dims(X_train, 1))
        Y_train = Tensor(np.expand_dims(Y_train, 1))
        ext_train = Tensor(ext_train)
        X_valid = Tensor(np.expand_dims(X_valid, 1))
        Y_valid = Tensor(np.expand_dims(Y_valid, 1))
        ext_valid = Tensor(ext_valid)
        X_test = Tensor(np.expand_dims(X_test, 1))
        Y_test = Tensor(np.expand_dims(Y_test, 1))
        ext_test = Tensor(ext_test)
        
        anchor_train = Tensor(np.expand_dims(anchor_train, 1))
        pos_train = Tensor(np.expand_dims(pos_train, 1))
        neg_train = Tensor(np.expand_dims(neg_train, 1))
        anchor_valid = Tensor(np.expand_dims(anchor_valid, 1))
        pos_valid = Tensor(np.expand_dims(pos_valid, 1))
        neg_valid = Tensor(np.expand_dims(neg_valid, 1))
        anchor_test = Tensor(np.expand_dims(anchor_test, 1))
        pos_test = Tensor(np.expand_dims(pos_test, 1))
        neg_test = Tensor(np.expand_dims(neg_test, 1))
        
        X8_train = Tensor(np.expand_dims(X8_train, 1))
        X8_valid = Tensor(np.expand_dims(X8_valid, 1))
        X8_test = Tensor(X8_test)
        
        data_train = torch.utils.data.TensorDataset(X_train, ext_train, Y_train)
        data_valid = torch.utils.data.TensorDataset(X_valid, ext_valid, Y_valid)
        data_test = torch.utils.data.TensorDataset(X_test, ext_test, Y_test)
        
        tc_train = torch.utils.data.TensorDataset(anchor_train, pos_train, neg_train)
        tc_valid = torch.utils.data.TensorDataset(anchor_valid, pos_valid, neg_valid)
        tc_test = torch.utils.data.TensorDataset(anchor_test, pos_test, neg_test)
        
        inf_train = torch.utils.data.TensorDataset(X8_train, ext_train, X_train)
        inf_valid = torch.utils.data.TensorDataset(X8_valid, ext_valid, X_valid)
        inf_test = torch.utils.data.TensorDataset(X8_test, ext_test, X_test)        
        datasets = {'train': data_train, 'val': data_valid, 'test':data_test}
        tc_datasets = {'train' : tc_train, 'val': tc_valid, 'test' : tc_test }
        inf_datasets = {'train': inf_train, 'val': inf_valid, 'test':inf_test}
        
        return datasets, tc_datasets, inf_datasets

    def setup_model(self):
        self.reg_model = reg_preTrain(self.config.reg_in_channels, self.config.reg_base_channels).to(self.device)
        self.reg_model.apply(weights_init_normal)
        torch.nn.utils.clip_grad_norm(self.reg_model.parameters(), max_norm=5.0)
        
        self.tc_model = tc_preTrain(self.config.tc_in_channels, self.config.tc_base_channels).to(self.device)
        self.tc_model.apply(weights_init_normal)
        torch.nn.utils.clip_grad_norm(self.tc_model.parameters(), max_norm=5.0)
        
        self.inf_model = inference_net().to(self.device)
        self.inf_model.apply(weights_init_normal)
        torch.nn.utils.clip_grad_norm(self.inf_model.parameters(), max_norm=5.0)

    def compose_dataset(self):
        datasets, tc_datasets, inf_datasets  = self.load_dataset()
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(len(datasets['val']) / self.config.test_batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        return datasets, tc_datasets, inf_datasets
    
    def compose_loader(self):
        datasets, tc_datasets, inf_datasets = self.compose_dataset()
        
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        test_dataset = datasets['test']
        
        tc_train_dataset = tc_datasets['train']
        tc_val_dataset = tc_datasets['val']
        tc_test_dataset = tc_datasets['test']
        
        inf_train_dataset = inf_datasets['train']
        inf_val_dataset = inf_datasets['val']
        inf_test_dataset = inf_datasets['test']
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Test [{}] Val [{}]'.format(toGreen(len(train_dataset)), toGreen(len(test_dataset)), toGreen(len(val_dataset))))
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        
        self.tc_train_loader = DataLoader(tc_train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.tc_val_loader = DataLoader(tc_val_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.tc_test_loader = DataLoader(tc_test_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        
        self.inf_train_loader = DataLoader(inf_train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.inf_val_loader = DataLoader(inf_val_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.inf_test_loader = DataLoader(inf_test_dataset, batch_size=self.config.test_batch_size, shuffle=False)


    def train(self):
        print(toGreen('\nSETUP TRAINING'))
        self.setup_train()
        if self.config.do_reg:
            print(toGreen('\nRegional-level Contrast Pre-training START'))
            for epoch in range(self.config.reg_epoch):
                total_loss, total_metrics = self.reg_train_epoch(epoch)
                avg_loss = total_loss / len(self.train_loader)
                avg_metrics = total_metrics / len(self.train_loader)
                self.logger.log_training(avg_loss, avg_metrics, epoch) 
                print_total('Reg-TRAIN', epoch, self.config.reg_epoch, 'Reg-Loss', avg_loss, self.config.metrics, avg_metrics)
                if epoch % self.config.valid_every_epoch == 0:
                    self.reg_validate(epoch, is_test=False)
            print(toGreen('\nRegional-level Contrast Pre-training END'))
            self.reg_validate(epoch, is_test=True)
            torch.save(self.reg_model.conv_pix.state_dict(), '../results/reg_pretrain_model')
        
        if self.config.do_tc:
            print(toGreen('\nTemporal contrastive Self-Supervison Pre-training START'))
            for epoch in range(self.config.tc_epoch):
                total_loss, total_metrics = self.tc_train_epoch(epoch)
                avg_loss = total_loss / len(self.tc_train_loader)
                avg_metrics = total_metrics / len(self.tc_train_loader)
                self.logger.log_training(avg_loss, avg_metrics, epoch) 
                print_total('TC-TRAIN', epoch, self.config.tc_epoch, 'TC-Loss', avg_loss, self.config.metrics, avg_metrics)
                if epoch % self.config.valid_every_epoch == 0:
                    self.tc_validate(epoch, is_test=False)
            print(toGreen('\nTemporal contrastive Self-Supervison Pre-training END'))
            self.tc_validate(epoch, is_test=True)
            torch.save(self.tc_model.conv_tc.state_dict(), '../results/tc_pretrain_model')
        
        if self.config.do_inf:
            print(toGreen('\nSpatial Super-resolution Inference Network Pre-training START'))
            for epoch in range(self.config.inf_epoch):
                total_loss, total_metrics = self.inf_train_epoch(epoch)
                avg_loss = total_loss / len(self.inf_train_loader)
                avg_metrics = total_metrics / len(self.inf_train_loader)
                self.logger.log_training(avg_loss, avg_metrics, epoch) 
                print_total('INF-TRAIN', epoch, self.config.inf_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)
                if epoch % self.config.valid_every_epoch == 0:
                    self.inf_validate(epoch, is_test=False)
            print(toGreen('\nSpatial Super-resolution Inference Network Pre-training END'))
            self.inf_validate(epoch, is_test=True)
            torch.save(self.inf_model.conv1.state_dict(), '../results/inf_pretrain_model')
        
        print(toGreen('\nPre-TRAINING END'))
        self.logger.close()
        
    def reg_validate(self, epoch, is_test=False):
        total_loss, total_metrics = self.reg_validate_epoch(epoch, is_test)
        avg_loss = total_loss / len(self.test_loader if is_test else self.val_loader)
        avg_metrics = total_metrics / len(self.test_loader if is_test else self.val_loader)
        self.logger.log_validation(avg_loss, avg_metrics, epoch)
        print_total('Reg-TEST' if is_test else 'Reg-VALID', epoch, self.config.reg_epoch, 'Reg-Loss', avg_loss, self.config.metrics, avg_metrics)


    def reg_train_epoch(self, epoch):
        self.reg_model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, ext, target) in enumerate(self.train_loader):
            data = data.to(self.device).float()
            self.reg_optimizer.zero_grad()

            output = self.reg_model(data)
            margin = self.config.reg_margin
            loss = loss_c(output, margin) 
            loss.backward()
            self.reg_optimizer.step()

            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = [0]
            total_metrics += this_metrics

            print_progress('Reg-TRAIN', epoch, self.config.reg_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, 'Reg-Loss', loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def reg_validate_epoch(self, epoch, is_test):
        self.reg_model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, ext, target) in enumerate(self.test_loader if is_test else self.val_loader):
            data = data.to(self.device).float()

            with torch.no_grad():
                output = self.reg_model(data)
            margin = self.config.reg_margin
            loss = loss_c(output, margin) 

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = [0]
            total_metrics += this_metrics


            print_progress('Reg-TEST' if is_test else 'Reg-VALID', epoch, self.config.reg_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                'Reg-Loss', loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics
    
    def tc_validate(self, epoch, is_test=False):
        total_loss, total_metrics = self.tc_validate_epoch(epoch, is_test)
        avg_loss = total_loss / len(self.tc_test_loader if is_test else self.tc_val_loader)
        avg_metrics = total_metrics / len(self.tc_test_loader if is_test else self.tc_val_loader)
        self.logger.log_validation(avg_loss, avg_metrics, epoch)
        print_total('TC-TEST' if is_test else 'TC-VALID', epoch, self.config.tc_epoch, 'TC-Loss', avg_loss, self.config.metrics, avg_metrics)


    def tc_train_epoch(self, epoch):
        self.tc_model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (anchor, pos, neg) in enumerate(self.tc_train_loader):
            anchor, pos, neg = anchor.to(self.device).float(), pos.to(self.device).float(), neg.to(self.device).float()
            self.tc_optimizer.zero_grad()

            anchor_tensor = self.tc_model(anchor)
            pos_tensor = self.tc_model(pos)
            neg_tensor = self.tc_model(neg)

            d_positive = distance_tensor(anchor_tensor, pos_tensor)
            d_negative = distance_tensor(anchor_tensor, neg_tensor)
            
            loss = torch.clamp(self.config.tc_margin + d_positive - d_negative, min=0.0).mean()
            loss.backward()
            self.reg_optimizer.step()

            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            this_metrics = [0]
            total_metrics += this_metrics

            print_progress('TC-TRAIN', epoch, self.config.tc_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, 'TC-Loss', loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def tc_validate_epoch(self, epoch, is_test):
        self.reg_model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (anchor, pos, neg) in enumerate(self.tc_test_loader if is_test else self.tc_val_loader):
            anchor, pos, neg = anchor.to(self.device).float(), pos.to(self.device).float(), neg.to(self.device).float()
            with torch.no_grad():
                anchor_tensor = self.tc_model(anchor)
                pos_tensor = self.tc_model(pos)
                neg_tensor = self.tc_model(neg)

            
            d_positive = distance_tensor(anchor_tensor, pos_tensor)
            d_negative = distance_tensor(anchor_tensor, neg_tensor)
            
            loss = torch.clamp(self.config.tc_margin + d_positive - d_negative, min=0.0).mean()

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()


            this_metrics = [0]
            total_metrics += this_metrics


            print_progress('TC-TEST' if is_test else 'TC-VALID', epoch, self.config.tc_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                'TC-Loss', loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics
    

    def inf_validate(self, epoch, is_test=False):
        total_loss, total_metrics = self.inf_validate_epoch(epoch, is_test)
        avg_loss = total_loss / len(self.inf_test_loader if is_test else self.inf_val_loader)
        avg_metrics = total_metrics / len(self.inf_test_loader if is_test else self.inf_val_loader)
        self.logger.log_validation(avg_loss, avg_metrics, epoch)
        print_total('INF-TEST' if is_test else 'INF-VALID', epoch, self.config.inf_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)


    def inf_train_epoch(self, epoch):
        self.inf_model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, ext, target) in enumerate(self.inf_train_loader):
            data, target = data.to(self.device).float(), target.to(self.device).float()
            self.inf_optimizer.zero_grad()

            output = self.inf_model(data)
            loss = self.loss(output, target) 
            loss.backward()
            self.inf_optimizer.step()

            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('INF-TRAIN', epoch, self.config.inf_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def inf_validate_epoch(self, epoch, is_test):
        self.reg_model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, ext, target) in enumerate(self.test_loader if is_test else self.val_loader):
            data, target = data.to(self.device).float(), target.to(self.device).float()

            with torch.no_grad():
                output = self.inf_model(data)
            loss = self.loss(output, target) 

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics


            print_progress('INF-TEST' if is_test else 'INF-VALID', epoch, self.config.tc_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics

    def setup_train(self):
        # loss, metrics, optimizer, scheduler
        try:
            loss_class = getattr(importlib.import_module('evaluation.metrics'), self.config.loss)
            self.loss = loss_class()
            self.metrics = [getattr(importlib.import_module('evaluation.metrics'), met)() for met in self.config.metrics]        
        except:
            print(toRed('No such metric in evaluation/metrics.py'))
            raise 

        try:
            # TODO Allow different types of optimizer like I did in scheduler 
            optim_class = getattr(importlib.import_module('torch.optim'), self.config.optimizer)
            self.reg_optimizer = optim_class(self.reg_model.parameters(), lr=1e-4)
            self.tc_optimizer = optim_class(self.tc_model.parameters(), lr=1e-4)
            self.inf_optimizer = optim_class(self.inf_model.parameters(), lr=1e-4)
        except:
            print(toRed('Error loading optimizer: {}'.format(self.config.optimizer)))
            raise 

        try: 
            scheduler_class = getattr(importlib.import_module('torch.optim.lr_scheduler'), self.config.scheduler)
            reg_scheduler_args = self.config.scheduler_args 
            reg_scheduler_args['optimizer'] = self.reg_optimizer
            self.reg_lr_scheduler = scheduler_class(**reg_scheduler_args)
            
            tc_scheduler_args = self.config.scheduler_args 
            tc_scheduler_args['optimizer'] = self.tc_optimizer
            self.tc_lr_scheduler = scheduler_class(**tc_scheduler_args)
            
            inf_scheduler_args = self.config.scheduler_args 
            inf_scheduler_args['optimizer'] = self.inf_optimizer
            self.inf_lr_scheduler = scheduler_class(**inf_scheduler_args)
        except:
            print(toRed('Error loading scheduler: {}'.format(self.config.scheduler)))
            raise 

        print_setup(self.config.loss, self.config.metrics, self.config.optimizer, self.config.scheduler)