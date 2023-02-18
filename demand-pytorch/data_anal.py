import numpy as np
import pickle

with open("dataset/bj-flow.pickle", "rb") as file:
    data = pickle.load(file)
flow = data['train']
n_timestamp, num_nodes, _ = flow.shape

timespan = (np.arange(n_timestamp) % 24) / 24
timespan = np.tile(timespan, (1, num_nodes, 1)).T
print(flow[:,:,1])
flow = np.concatenate((flow, timespan), axis=2)
mask = np.sum(flow, axis=(1,2)) > 5000
input_len =12
output_len = 3
feature, data, label  = [], [], []
for i in range(n_timestamp - input_len - output_len + 1):
    if mask[i + input_len: i + input_len + output_len].sum() != output_len:
        continue
    data.append(flow[i: i + input_len])
    label.append(flow[i + input_len: i + input_len + output_len])

print(data[0].shape, label[0].shape)

data = np.array(np.stack(data)) # [B, T, N, D]
label = np.array(np.stack(label)) # [B, T, N, D]

print(data.shape, label.shape)