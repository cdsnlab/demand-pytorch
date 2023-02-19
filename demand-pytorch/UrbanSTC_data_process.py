import numpy as np
import os
import math
import torch
import pickle
from torch.utils.data import DataLoader
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def create_tc_data_HardSample(type, mode):
    datapath = os.path.join('dataset', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'X.npy'))

    anchor = []
    p1 = []
    p2 = []

    length = len(A)
    for i in range(length):
        maxn = -float('inf')
        minn = float('inf')
        kmax, kmin = i, i
        for j in range(length):
            if i == j:
                continue
            dist = np.sqrt(np.sum(np.square(A[i] - A[j])))

            if dist > maxn:
                maxn = dist
                kmax = j

            if dist < minn:
                minn = dist
                kmin = j

        # print(minn, maxn, i, kmin, kmax)

        anchor.append(A[i])
        p1.append(A[kmin])
        p2.append(A[kmax])

    anchor = np.array(anchor)
    p1 = np.array(p1)
    p2 = np.array(p2)

    anchor_path = os.path.join(datapath, 'anchor.npy')
    pos_path = os.path.join(datapath, 'pos.npy')
    neg_path = os.path.join(datapath, 'neg.npy')

    np.save(anchor_path, anchor)
    np.save(pos_path, p1)
    np.save(neg_path, p2)

def create_scaler_data(type, mode):
    up_size = 4
    if mode == 'BikeNYC':
        up_size = 2

    datapath = os.path.join('dataset', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'X.npy'))
    (x, y, z) = A.shape
    zeros = np.zeros(shape=(x, y//up_size, z//up_size))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = i//up_size
                jj = j//up_size
                zeros[k][ii][jj] += A[k][i][j]

    if mode == 'BikeNYC':
        temp_path = os.path.join(datapath, '10X.npy')
        np.save(temp_path, zeros)
    else:
        temp_path = os.path.join(datapath, '8X.npy')
        np.save(temp_path, zeros)

create_scaler_data(type='train', mode='P1')
create_scaler_data(type='valid', mode='P1')
create_scaler_data(type='test', mode='P1')
create_tc_data_HardSample(type='train', mode='P1')
create_tc_data_HardSample(type='valid', mode='P1')
create_tc_data_HardSample(type='test', mode='P1')
