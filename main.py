# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf
import os
import numpy as np

import datetime as dt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)
tf.set_random_seed(42)
np.random.seed(42)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train, model_train_cls
from models.tester import dataErrorMap, model_test, model_test_cls

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('graph', type=str, default='de')
parser.add_argument('feature', type=str, default='de')
parser.add_argument('interval', type=int)
parser.add_argument('--daySlot', type=int, default=240)
# parser.add_argument('--n_route', type=int, default=297)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=1000)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--inf_mode', type=str, default='sep')
parser.add_argument('--cls', type=int, default=4)

args = parser.parse_args()
print(f'Training configs: {args}')

# n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]
outputPath = f"./output/{args.cls}/models/"

# Load wighted adjacency matrix W
W = weight_matrix(args.graph)
n = W.shape[0]
n_his, n_pred = args.n_his, args.n_pred

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
n_train, n_val, n_test = 11, 5, 5
PeMS = data_gen(args.feature, (n_train, n_val, n_test), n, n_his + n_pred, args.interval, args.cls)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    if args.cls < 2:
        model_train(PeMS, blocks, args)
        model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
    else:
        trainAcc, valAcc, testAcc, cpAcc = model_train_cls(PeMS, blocks, args, outputPath)
        model_test_cls(PeMS, args.batch_size, n_his, n_pred, outputPath)
        dataErrorMap(PeMS, args.batch_size, n_his, n_pred, outputPath)
        with open(pjoin(outputPath, "log.txt"), "a") as f:
            if f.tell() == 0:
                f.write(f"Time, Epoch, CLS, BatchSize, LR, Train Acc, Val Acc, Test Acc, Copy Acc, Data Path\n")
            f.write(f"{dt.datetime.today()}, {args.epoch}, {args.cls}, {args.batch_size}, {args.lr}, {trainAcc}, {valAcc}, {testAcc}, {cpAcc}, {args.feature}\n")

