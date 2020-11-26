# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from tensorflow.python.ops.gen_batch_ops import batch
from utils.math_utils import z_score

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats, label):
        self.__data = data
        self.label = label
        self.stats = stats
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_label(self, type):
        return self.label[type]

    def get_stats(self):
        return self.stats
        # return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])
    
    def get_feature_num(self, type):
        return self.__data[type].shape[2]

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (pad_size, 0)

    return np.pad(array, pad_width=npad, mode='edge')

def m_seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    n_slot = day_slot * len_seq - n_frame + 1
    tmp_seq = np.zeros((n_slot, n_frame, n_route, C_0))
    for i in range(n_slot):
        sta = i + offset * day_slot
        end = sta + n_frame
        # print(data_seq.shape, i, sta, end)
        tmp_seq[i, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    tmp_seq = pad_along_axis(tmp_seq, 11, 1)
    return tmp_seq


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def data_gen(file_path, data_config, n_route, n_frame, interval, cls):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values.transpose()
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
        exit()
    day_slot = 24 * 60 // interval
    dataStartIdx = 6 * 60 // interval
    dataEndIdx = 18 * 60 // interval
    data_seq = data_seq[dataStartIdx:-dataEndIdx, :]
    label, labelCenter, dBin = getLabel(data_seq, classNum=cls)

    seq_train = m_seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = m_seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = m_seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)
    label_train = m_seq_gen(n_train, label, 0, n_frame, n_route, day_slot)
    label_val = m_seq_gen(n_val, label, n_train, n_frame, n_route, day_slot)
    label_test = m_seq_gen(n_test, label, n_train + n_val, n_frame, n_route, day_slot)
    print(seq_train[0, :, 0, 0])
    print(np.unique(label_train))

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train), 'center': labelCenter, 'bin': dBin}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    label = {'train': label_train, 'val': label_val, 'test': label_test}
    dataset = Dataset(x_data, x_stats, label)
    return dataset


def getLabel(data, classNum=4, binNum=256, cutoffSigma=3):
    x = np.copy(data)
    mean = np.average(x)
    std = np.std(x)
    x[x < mean - 3 * std] = mean - cutoffSigma * std
    x[x > mean + 3 * std] = mean + cutoffSigma * std
    n, bin = np.histogram(x, binNum)
    dx = np.digitize(x, bin) - 1
    lut = np.zeros(binNum + 1)
    lut[1:] = np.cumsum(n)
    lut = (lut / lut[-1] * classNum).astype(np.int)
    lut[lut > (classNum-1)] = classNum - 1
    label = lut[dx]
    labelCenter = np.zeros(classNum)
    dBin = np.zeros(classNum + 1)
    dBin[0] = bin[0]
    dBin[1:] = [bin[lut == i][-1] for i in range(classNum)]
    labelCenter = [np.average(bin[lut == i]) for i in range(classNum)]
    print(f"Label Center: {labelCenter}")
    labelCount = np.bincount(label.reshape(-1))
    labelDist = labelCount / np.sum(labelCount)
    print(f"Label Distribution: {labelDist}")
    return label, labelCenter, dBin


def gen_batch(inputs, batch_size, label=None, dynamic_batch=False, shuffle=False, roll=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, 1 if roll else batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        if label is not None:
            yield inputs[slide], label[slide]
        else:
            yield inputs[slide]
