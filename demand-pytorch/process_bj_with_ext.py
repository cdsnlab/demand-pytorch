import h5py
import numpy as np
import pickle 
import time
from copy import copy


class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

def load_holiday(timeslots):
    """
    :param timeslots:
    :param fname:
    :return:
    [[1],[1],[0],[0],[0]...]
    """
    f = open('bj/BJ_Holiday.txt', 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    return H[:, None]  # to 2d

def load_meteorol(timeslots):
    f = h5py.File('bj/BJ_Meteorology.h5', 'r')
    Timeslot = np.array(f['date'])
    WindSpeed = np.array(f['WindSpeed'])
    Weather = np.array(f['Weather'])
    Temperature = np.array(f['Temperature'])
    f.close()

    M = dict()  # map timeslot to index

    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature

    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())


    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    return merge_data

def remove_incomplete_days(data, timestamps, T=48):
    """
    remove a certain day which has not 48 timestamps
    :param data:
    :param timestamps:
    :param T:
    :return:
    """

    days = []  # available days: some day only contain some seqs
    days_incomplete = [] # incomplete (does not have 48 slots) days
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def load_stdata(fname):
    """
    split the data and date(timestamps)
    :param fname:
    :return:
    """
    f = h5py.File(fname, 'r')
    data = np.array(f['data'])
    timestamps = np.array(f['date'])
    f.close()
    return data, timestamps

def timestamp2vec(timestamps):
    """
    :param timestamps:
    :return:
    exampel:
    [b'2018120505', b'2018120106']
    #[[0 0 1 0 0 0 0 1]  
     [0 0 0 0 0 1 0 0]]  
    """
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8],encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    # vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


data_all = []
timestamps_all = []
for year in range(13, 17): ## todo change 14 to 17 after download is complete
    fname = 'bj/BJ{}_M32x32_T30_InOut.h5'.format(year)
    data, timestamps = load_stdata(fname)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps)
    data[data < 0] = 0.
    data_all.append(data)
    timestamps_all.append(timestamps)

data_all = np.concatenate(data_all, axis=0)
timestamps_all = np.concatenate(timestamps_all, axis=0)


meta_feature = []

holiday_feature = load_holiday(timestamps_all)
meteorol_feature = load_meteorol(timestamps_all)
time_feature = timestamp2vec(timestamps_all)

meta_feature.append(time_feature)
meta_feature.append(holiday_feature)
meta_feature.append(meteorol_feature)
meta_feature = np.hstack((meta_feature))


n_timestamp = data_all.shape[0]
num_train = int(n_timestamp * 0.7)
num_val = int(n_timestamp * 0.2)
num_test = n_timestamp - num_train - num_val

data_train = np.vstack(copy(data_all))[:num_train]

mmn = MinMaxNormalization()
mmn.fit(data_train)
data_all = [mmn.transform(d) for d in data_all]
data_all = np.asarray(data_all)


train_flow, val_flow, test_flow = data_all[:num_train], data_all[num_train: num_train + num_val], data_all[-num_test:]
train_ext, val_ext, test_ext = meta_feature[:num_train], meta_feature[num_train: num_train + num_val], meta_feature[-num_test:]
train_date, val_date, test_date = timestamps_all[:num_train], timestamps_all[num_train: num_train + num_val], timestamps_all[-num_test:]

bj_data = {
    "train": (train_flow, train_ext, train_date),
    "val": (val_flow, val_ext, val_date),
    "test": (test_flow, test_ext, test_date), 
    "data_min": mmn._min,
    "data_max": mmn._max,
}

with open("bj/bj-flow-ext.pickle","wb") as fw:
    pickle.dump(bj_data, fw)