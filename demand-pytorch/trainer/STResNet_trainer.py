import math 
import time
import numpy as np 
import pandas as pd
from datetime import datetime
from data.utils import *
from data.datasets import STResNetDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger

class STResNetTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger(self.save_name)
        self.len_closeness = self.config.len_closeness
        self.len_period = self.config.len_period
        self.len_trend = self.config.len_trend
        self.PeriodInterval = self.config.PeriodInterval
        self.TrendInterval = self.config.TrendInterval
        self.T = self.config.T
    
    def string2timestamp(self, strings, T=48):
        """
        :param strings:
        :param T:
        :return:
        example:
        str = [b'2013070101', b'2013070102']
        print(string2timestamp(str))
        [Timestamp('2013-07-01 00:00:00'), Timestamp('2013-07-01 00:30:00')]
        """
        timestamps = []

        time_per_slot = 24.0 / T
        num_per_T = T // 24
        for t in strings:
            year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
            timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                    minute=(slot % num_per_T) * int(60.0 * time_per_slot))))
        return timestamps

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def load_dataset(self):
        with open(os.path.join(self.config.dataset_dir, self.config.dataset_name+".pickle"), "rb") as file:
            data = pickle.load(file)
        
        self.data_min = data['data_min']
        self.data_max = data['data_max']

        datasets = {}

        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        for category in ['train', 'val', 'test']:
            XC = []
            XP = []
            XT = []
            E = []
            Y = []

            self.data = data[category][0]

            pd_timestamps = self.string2timestamp(data[category][2])

            self.get_index = dict()
            for i, ts in enumerate(pd_timestamps):
                self.get_index[ts] = i
            
            depends = [range(1, self.len_closeness + 1),
                   [self.PeriodInterval * self.T * j for j in range(1, self.len_period + 1)],
                   [self.TrendInterval * self.T * j for j in range(1, self.len_trend + 1)]]
            
            i = max(self.T * self.TrendInterval * self.len_trend, self.T * self.PeriodInterval * self.len_period, self.len_closeness)
            while i < len(pd_timestamps):
                Flag = True
                for depend in depends:
                    if Flag is False:
                        break
                    Flag = self.check_it([pd_timestamps[i] - j * offset_frame for j in depend])

                if Flag is False:
                    i += 1
                    continue

                x_c = [self.get_matrix(pd_timestamps[i] - j * offset_frame) for j in depends[0]]
                x_p = [self.get_matrix(pd_timestamps[i] - j * offset_frame) for j in depends[1]]
                x_t = [self.get_matrix(pd_timestamps[i] - j * offset_frame) for j in depends[2]]
                e = data[category][1][self.get_index[pd_timestamps[i]]]
                y = self.get_matrix(pd_timestamps[i])

                if self.len_closeness > 0:
                    XC.append(np.vstack(x_c))
                if self.len_period > 0:
                    XP.append(np.vstack(x_p))
                if self.len_trend > 0:
                    XT.append(np.vstack(x_t))
                Y.append(y)
                E.append(e)
                i += 1
            
            XC = np.asarray(XC)
            XP = np.asarray(XP)
            XT = np.asarray(XT)
            E = np.asarray(E)
            Y = np.asarray(Y)

            x = (XC, XP, XT, E)
            y = Y

            datasets[category] = {'x': x, 'y': y}
        return datasets

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = STResNetDataset(datasets[category])
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
    
    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self.data_max - self.data_min) + self.data_min
        return X

    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.train_loader):
            XC = data[0].to(self.device).float()
            XP = data[1].to(self.device).float()
            XT = data[2].to(self.device).float()
            E = data[3].to(self.device).float()
            target = target.to(self.device).float()

            self.optimizer.zero_grad()

            output = self.model(XC, XP, XT, E)

            # rescale
            output = self.inverse_transform(output)
            target = self.inverse_transform(target)

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

        for batch_idx, (data, target) in enumerate(self.test_loader if is_test else self.val_loader):
            XC = data[0].to(self.device).float()
            XP = data[1].to(self.device).float()
            XT = data[2].to(self.device).float()
            E = data[3].to(self.device).float()
            target = target.to(self.device).float()

            with torch.no_grad():
                output = self.model(XC, XP, XT, E)

            #rescale
            output = self.inverse_transform(output)
            target = self.inverse_transform(target)

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