import numpy as np
import pickle

with open("dataset/bj-flow.pickle", "rb") as file:
    data = pickle.load(file)
flow = data['edge']
print(flow.shape)
