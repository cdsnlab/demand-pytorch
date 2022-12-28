import scipy.linalg as la
import numpy as np
import pickle
with open("../datasets/bj-flow.pickle", "rb") as file:
    data = pickle.load(file)
def make_matrix(rows, cols):
    n = rows*cols
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            if c > 0: 
                M[i-1,i] = M[i,i-1] = 1
            if c < cols-1:
                M[i+1,i] = M[i,i+1] = 1
            if r > 0: 
                M[i-cols,i] = M[i,i-cols] = 1
            if r < rows-1:
                M[i+cols,i] = M[i+cols,i] = 1
            if c > 0 and r > 0:
                M[i-cols-1,i] = M[i,i-cols-1] = 1
            if c < cols-1 and r > 0:
                M[i-cols+1,i] = M[i,i-cols+1] = 1
            if c > 0 and r < rows-1:
                M[i+cols-1,i] = M[i,i+cols-1] = 1
            if c < cols-1 and r < rows-1:
                M[i+cols+1,i] = M[i,i+cols+1] = 1
    return M
adj = make_matrix(32, 32)
data['adj'] = adj 

with open("../datasets/bj-flow.pickle", "wb") as file:
    pickle.dump(data, file)