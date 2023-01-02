import math 
import time
import numpy as np 
from data.utils import *
from data.datasets import DeepSTNDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger
from torchsummary import summary as summary_

class MM:
    def __init__(self,MM_max,MM_min):
        self.max=MM_max
        self.min=MM_min    

def lzq_load_data(len_test,len_closeness,len_period,len_trend,T_closeness=1,T_period=24,T_trend=24*7):
    
    all_data=np.load('data/flow_data.npy')
    len_total,feature,map_height,map_width=all_data.shape
    #all_data=np.arange(48*24*7*256).reshape(-1,2,16,8)
    #len_total,feature,map_height,map_width=all_data.shape
    print('all_data shape: ',all_data.shape)
    mm=MM(np.max(all_data),np.min(all_data))
    print('max=',mm.max,' min=',mm.min)
    
    #for time
    time=np.arange(len_total,dtype=int)
    #hour
    time_hour=time%T_period
    matrix_hour=np.zeros([len_total,24,map_height,map_width])
    for i in range(len_total):
        matrix_hour[i,time_hour[i],:,:]=1
    #day
    time_day=(time//T_period)%7
    matrix_day=np.zeros([len_total,7,map_height,map_width])
    for i in range(len_total):
        matrix_day[i,time_day[i],:,:]=1
    #con
    matrix_T=np.concatenate((matrix_hour,matrix_day),axis=1)
    
    all_data=(2.0*all_data-(mm.max+mm.min))/(mm.max-mm.min)
    print('mean=',np.mean(all_data),' variance=',np.std(all_data))
    
    if len_trend>0:
        number_of_skip_hours=T_trend*len_trend
    elif len_period>0:
        number_of_skip_hours=T_period*len_period
    elif len_closeness>0:
        number_of_skip_hours=T_closeness*len_closeness  
    else:
        print("wrong")
    print('number_of_skip_hours:',number_of_skip_hours)
    
    Y=all_data[number_of_skip_hours:len_total]

    if len_closeness>0:
        X_closeness=all_data[number_of_skip_hours-T_closeness:len_total-T_closeness]
        for i in range(len_closeness-1):
            X_closeness=np.concatenate((X_closeness,all_data[number_of_skip_hours-T_closeness*(2+i):len_total-T_closeness*(2+i)]),axis=1)
    if len_period>0:
        X_period=all_data[number_of_skip_hours-T_period:len_total-T_period]
        for i in range(len_period-1):
            X_period=np.concatenate((X_period,all_data[number_of_skip_hours-T_period*(2+i):len_total-T_period*(2+i)]),axis=1)
    if len_trend>0:
        X_trend=all_data[number_of_skip_hours-T_trend:len_total-T_trend]
        for i in range(len_trend-1):
            X_trend=np.concatenate((X_trend,all_data[number_of_skip_hours-T_trend*(2+i):len_total-T_trend*(2+i)]),axis=1)
    
    matrix_T=matrix_T[number_of_skip_hours:]
    
    X_closeness_train=X_closeness[:-len_test] 
    X_period_train=X_period[:-len_test] 
    X_trend_train=X_trend[:-len_test]  
    T_train=matrix_T[:-len_test] 
    X_closeness_test=X_closeness[-len_test:] 
    X_period_test=X_period[-len_test:] 
    X_trend_test=X_trend[-len_test:]          
    T_test=matrix_T[-len_test:]         
    
    # X_train=[X_closeness_train,X_period_train,X_trend_train]
    # X_test=[X_closeness_test,X_period_test,X_trend_test]
    X_train=np.concatenate((X_closeness_train,X_period_train,X_trend_train),axis=1)
    X_test=np.concatenate((X_closeness_test,X_period_test,X_trend_test),axis=1)
    Y_train=Y[:-len_test] 
    Y_test=Y[-len_test:] 

    len_train=X_closeness_train.shape[0]
    len_test=X_closeness_test.shape[0]
    print('len_train='+str(len_train))
    print('len_test ='+str(len_test ))
    
    poi=np.load('data/poi_data.npy')
    for i in range(poi.shape[0]):
        poi[i]=poi[i]/np.max(poi[i])
    P_train=np.repeat(poi.reshape(1,poi.shape[0],map_height,map_width),len_train,axis=0)
    P_test =np.repeat(poi.reshape(1,poi.shape[0],map_height,map_width),len_test ,axis=0)
    return X_train,T_train,P_train,Y_train,X_test,T_test,P_test,Y_test,mm.max-mm.min

class DeepSTNTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.logger = Logger(self.save_name)
    
    def load_dataset(self):
        data = np.load(self.config.dataset_dir , "/" , self.config.dataset_name+".npy")
        datasets = {}
        for category in ['train', 'val', 'test']:
            x, y = seq2instance(data[category].reshape(data[category].shape[0], -1), self.config.num_his, self.config.num_pred)
            if category == 'train':
                self.mean, self.qqstd = np.mean(x), np.std(x)
            x = (x - self.mean) / self.std 
            datasets[category] = {'x': x, 'y': y}
        return datasets

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = DeepSTNDataset(datasets[category])
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(len(datasets['val']) / self.config.test_batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        return datasets['train'], datasets['val'], datasets['test']
    
    def compose_loader(self):
        train_dataset,T_train,P_train,train_target,test_dataset,T_test,P_test,val_dataset,MM = lzq_load_data(24*14,3,4,4)
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Test [{}] Val [{}]'.format(toGreen(len(train_dataset)), toGreen(len(test_dataset)), toGreen(len(val_dataset))))
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.target = train_target

    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.train_loader):
            target = self.target
            self.optimizer.zero_grad()

            output = self.model(data)
            summary_(self.model, (22, 21, 12))
            
            output = output[:, -self.config.num_pred:]
            output = output * self.std 
            output = output + self.mean

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
            data, target = data.to(self.device).float(), target.to(self.device).float()

            with torch.no_grad():
                output = self.model(data)
            output = output[:, -self.config.num_pred:]
            output = output * self.std 
            output = output + self.mean

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


