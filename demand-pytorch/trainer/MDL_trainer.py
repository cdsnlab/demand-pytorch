import math 
import time
import numpy as np 
import pandas as pd
from data.utils import *
from data.datasets import MDLDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger

class MDLTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger(self.save_name)
        self.T = config.T

    def load_dataset(self):
        with open(os.path.join(self.config.dataset_dir, self.config.dataset_name+".pickle"), "rb") as file:
            data = pickle.load(file)

        datasets = {}

        #self.node_input_scaler = MinMax01Scaler(np.min(data['train'][0]), np.max(data['train'][0]))
        self.node_output_scaler = MinMax11Scaler(np.min(data['train'][0]), np.max(data['train'][0]))
        #self.edge_input_scaler = MinMax01Scaler(np.min(data['train'][1]), np.max(data['train'][1]))
        self.edge_output_scaler = MinMax11Scaler(np.min(data['train'][1]), np.max(data['train'][1]))

        for category in ['train', 'val', 'test']:
            len_trend, len_period, len_closeness = self.config.len_trend, self.config.len_period, self.config.len_closeness
            start = max(self.T * 7 * self.config.len_trend, max(24 * len_period, len_closeness))

            XT, XP, XC, XY = [], [], [], []
            MT, MP, MC, MY = [], [], [], []
            E = []

            for i in range(start, len(data[category][0])):
                len1, len2, len3 = len_trend, len_period, len_closeness
                xt, xp, xc, xy = [], [], [], []
                mt, mp, mc, my = [], [], [], []

                while len1 > 0:
                    node_f = data[category][0][i - self.T * 7 * len1]
                    edge_f = data[category][1][i - self.T * 7 * len1]
                    xt.append(node_f)
                    mt.append(edge_f)
                    len1 = len1 - 1
                while len2 > 0:
                    node_f = data[category][0][i - self.T * len2]
                    edge_f = data[category][1][i - self.T * len2]
                    xp.append(node_f)
                    mp.append(edge_f)
                    len2 = len2 - 1
                while len3 > 0:
                    node_f = data[category][0][i - len3]
                    edge_f = data[category][1][i - len3]
                    xc.append(node_f)
                    mc.append(edge_f)
                    len3 = len3 - 1 

                xy = data[category][0][i:i+1]
                my = data[category][1][i:i+1]

                XT.append(xt)
                XP.append(xp)
                XC.append(xc)
                MT.append(mt)
                MP.append(mp)
                MC.append(mc)

                XY.append(xy)
                MY.append(my)

                #e = np.random.rand(self.config.external_dim)
                e = np.zeros(self.config.external_dim)
                E.append(e)
                     
            
            #XT = self.node_input_scaler.transform(np.array(XT))
            #XP = self.node_input_scaler.transform(np.array(XP))
            #XC = self.node_input_scaler.transform(np.array(XC))
            XT = np.array(XT)
            XP = np.array(XP)
            XC = np.array(XC)
            XY = self.node_output_scaler.transform(np.array(XY))

            #MT = self.edge_input_scaler.transform(np.array(MT))
            #MP = self.edge_input_scaler.transform(np.array(MP))
            #MC = self.edge_input_scaler.transform(np.array(MC))
            MT = np.array(MT)
            MP = np.array(MP)
            MC = np.array(MC)
            MY = self.edge_output_scaler.transform(np.array(MY))

            E = np.array(E)

            x = (XT, XP, XC, MT, MP, MC, E)
            y = (XY, MY)

            print(np.max(XT), np.min(XP))


            datasets[category] = {'x': x, 'y': y}
        return datasets

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = MDLDataset(datasets[category])
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
            XT = data[0].to(self.device).float()
            XP = data[1].to(self.device).float()
            XC = data[2].to(self.device).float()
            MT = data[3].to(self.device).float()
            MP = data[4].to(self.device).float()
            MC = data[5].to(self.device).float()
            E = data[6].to(self.device).float()
            XY = target[0].to(self.device).float()
            MY = target[1].to(self.device).float()

            self.optimizer.zero_grad()

            X = [XT, XP, XC]
            M = [MT, MP, MC]


            b, t, c, h, w = XY.shape
            XY = XY.reshape((b, -1, h, w))
            MY = MY.reshape((b, -1, h, w))

            XY = self.node_output_scaler.inverse_transform(XY)
            MY = self.edge_output_scaler.inverse_transform(MY)
            
            train_loss, node_output, edge_output = self.model.multask_loss(X, M, E, XY, MY, self.node_output_scaler, self.edge_output_scaler)

            train_loss.backward()
            self.optimizer.step()

            # print("loss: ", train_loss)

            training_time = time.time() - start_time
            start_time = time.time()

            with torch.no_grad():
                node_output, edge_output = self.model(X, M, E)

            node_output = self.node_output_scaler.inverse_transform(node_output)
            edge_output = self.edge_output_scaler.inverse_transform(edge_output)

            # print(torch.min(XC), torch.max(XC))
            # print(torch.min(node_output), torch.max(node_output))
            # print(torch.min(XY), torch.max(XY))
            # print("mean start:")
            # print(node_output.shape, XY.shape)
            # print(torch.mean(torch.square(node_output - XY)))
            # print("mean end")
            # print()


            loss = self.loss(node_output, XY)

            total_loss += loss.item()

            node_output = node_output.detach().cpu()
            XY = XY.detach().cpu()
            # edge_output = edge_output.detach().cpu()
            # MY = MY.detach().cpu()
            #output = output.detach().cpu()
            #target = target.detach().cpu()

            # this_metrics = self._eval_metrics(edge_output, MY)
            this_metrics = self._eval_metrics(node_output, XY)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def validate_epoch(self, epoch, is_test):
        self.model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.test_loader if is_test else self.val_loader):
            XT = data[0].to(self.device).float()
            XP = data[1].to(self.device).float()
            XC = data[2].to(self.device).float()
            MT = data[3].to(self.device).float()
            MP = data[4].to(self.device).float()
            MC = data[5].to(self.device).float()
            E = data[6].to(self.device).float()
            XY = target[0].to(self.device).float()
            MY = target[1].to(self.device).float()

            X = [XT, XP, XC]
            M = [MT, MP, MC]

            with torch.no_grad():
                node_output, edge_output = self.model(X, M, E)
            
            node_output = self.node_output_scaler.inverse_transform(node_output)
            edge_output = self.edge_output_scaler.inverse_transform(edge_output)
            XY = self.node_output_scaler.inverse_transform(XY)
            MY = self.edge_output_scaler.inverse_transform(MY)

            loss = self.loss(node_output, XY)

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            node_output = node_output.detach().cpu()
            XY = XY.detach().cpu()

            #this_metrics = self._eval_metrics(output, target)
            this_metrics = self._eval_metrics(node_output, XY)
            total_metrics += this_metrics

            print_progress('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics

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