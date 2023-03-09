import h5py
import numpy as np
import pickle 
import time
import torch
from copy import copy


file_path = 'datasets/nyc'

def get_X(M):
    s = M.shape

    X = np.zeros((s[0], 2, s[2], s[3]))

    for idx in range(s[0]):
        for i in range(s[2]):
            for j in range(s[3]):
                X[idx][0][i][j] = np.sum(M[idx][0][i][j])
                X[idx][1][i][j] = np.sum(M[idx][1][i][j])
    return X

def load_flow():
    # (2, 1920, 10, 20, 10, 20)
    # M_train: edge inflow/outflow train data
    M_train = np.load(file_path + '/bike_flow_train.npz')["flow"]
    M_train = np.transpose(M_train, (1, 0, 2, 3, 4, 5)) # (1920, 2, 10, 20, 10, 20)
    print("train: ", M_train.shape)

    M_test = np.load(file_path + '/bike_flow_test.npz')["flow"]
    M_test = np.transpose(M_test, (1, 0, 2, 3, 4, 5))
    print("test: ", M_test.shape)
    
    M = np.concatenate([M_train, M_test], axis=0)

    X = get_X(M)
    print(X.shape) # (2880, 2, 10, 20)

    M = np.reshape(M, (M.shape[0], -1, M.shape[4], M.shape[5]))
    print(M.shape) # (2880, 400, 10, 20)

    return X, M


X, M = load_flow()

print(X[:3])

n = X.shape[0]
num_train = int(n * 0.6)
num_val = int(n * 0.2)
num_test = n - num_train - num_val

X_train, X_val, X_test = X[:num_train], X[num_train : num_train+num_val], X[-num_test:]
M_train, M_val, M_test = M[:num_train], M[num_train : num_train+num_val], M[-num_test:]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
data = {
    "train": (X_train, M_train),
    "val": (X_val, M_val),
    "test": (X_test, M_test),
}

with open("datasets/nyc/nyc-taxi-flow.pickle","wb") as fw:
    pickle.dump(data, fw)
