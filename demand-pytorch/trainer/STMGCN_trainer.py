import math 
import time
import numpy as np 
from data.utils import *
from data.datasets import STMGCNDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
from logger.logger import Logger

class STMGCNTrainer(BaseTrainer):
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
        for category in ['train', 'val', 'test']:
            x, y = seq2instance(data[category].reshape(data[category].shape[0], -1), self.config.num_his, self.config.num_pred, offset=self.config.offset)
            if category == 'train':
                self.mean, self.std = np.mean(x), np.std(x)
            x = (x - self.mean) / self.std 
            datasets[category] = {'x': x, 'y': y}
        adj = torch.tensor(data['adj'])
        adj_proc = Adj_Preprocessor(**self.config.sta_kernel_config)
        self.adj = adj_proc.process(adj)
        return datasets

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'val', 'test']:
            datasets[category] = STMGCNDataset(datasets[category])
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
            data, target, adj = data.to(self.device).float(), target.to(self.device).float(), self.adj.to(self.device).float()
            self.optimizer.zero_grad()

            output = self.model(data, [adj])
            output = output * self.std 
            output = output + self.mean
            output = output.unsqueeze(1)

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
            data, target, adj = data.to(self.device).float(), target.to(self.device).float(), self.adj.to(self.device).float()

            with torch.no_grad():
                output = self.model(data, [adj])
            output = output * self.std 
            output = output + self.mean
            output = output.unsqueeze(1)

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


class Adj_Preprocessor(object):
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, adj:torch.Tensor):
        '''
        Generate adjacency matrices
        :param adj: input adj matrix - (N, N) torch.Tensor
        :return: processed adj matrix - (K_supports, N, N) torch.Tensor
        '''
        kernel_list = list()

        if self.kernel_type in ['localpool', 'chebyshev']:  # spectral
            adj_norm = self.symmetric_normalize(adj)
            # adj_norm = self.random_walk_normalize(adj)     # for asymmetric normalization
            if self.kernel_type == 'localpool':
                localpool = torch.eye(adj_norm.shape[0]) + adj_norm  # same as add self-loop first
                kernel_list.append(localpool)

            else:  # chebyshev
                laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                rescaled_laplacian = self.rescale_laplacian(laplacian_norm)
                kernel_list = self.compute_chebyshev_polynomials(rescaled_laplacian, kernel_list)

        elif self.kernel_type == 'random_walk_diffusion':  # spatial

            # diffuse k steps on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)
            '''
            # diffuse k steps bidirectionally on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            P_backward = self.random_walk_normalize(adj.T)
            forward_series, backward_series = [], []
            forward_series = self.compute_chebyshev_polynomials(P_forward.T, forward_series)
            backward_series = self.compute_chebyshev_polynomials(P_backward.T, backward_series)
            kernel_list += forward_series + backward_series[1:]  # 0-order Chebyshev polynomial is same: I
            '''
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')

        # print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
        kernels = torch.stack(kernel_list, dim=0)

        return kernels

    @staticmethod
    def random_walk_normalize(A):   # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)   # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.
        D = torch.diag(d_inv)
        A_norm = torch.mm(D, A)
        return A_norm

    @staticmethod
    def symmetric_normalize(A):
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = torch.mm(torch.mm(D, A), D)
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
        try:
            lambda_ = torch.eig(L)[0][:,0]      # get the real parts of eigenvalues
            lambda_max = lambda_.max()      # get the largest eigenvalue
        except:
            print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescale = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescale

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        return T_k

