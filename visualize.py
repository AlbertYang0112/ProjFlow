import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import imageio
import matplotlib.ticker as ticker


def block_vis():
    df = pd.read_csv("data/25.csv")
    df = df.sample(frac=0.01)
    plt.grid()
    min_x = np.min(df['x'])
    max_x = np.max(df['x'])
    min_y = np.min(df['y'])
    max_y = np.max(df['y'])
    plt.scatter((df['x'] - min_x) / (max_x - min_x), (df['y'] - min_y) / (max_y - min_y), s=5)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('x coordinate (normalized)')
    plt.ylabel('y coordinate (normalized)')
    plt.show()


def pin_num():
    df = pd.read_csv("data/25.csv")
    pin_count = df.loc[:, 'advertiser_id'].value_counts()
    print(pin_count)
    w = Counter(list(pin_count))
    print(w)
    plt.scatter(w.keys(), w.values(), s=1)
    plt.xlabel('# of pins per user')
    plt.ylabel('user count')
    plt.show()


def pin_dyn_dis():
    input_path = 'data/March/processed/'
    plt.ion()
    sns.set_context({"figure.figsize": (7, 5)})
    name = [str(i).zfill(2) for i in range(1, 31)]
    # for i in range(1, 6):
    #     name.append(str(i).zfill(2))
    image_list = []
    for index in range(len(name)):
        plt.clf()
        mat = np.load(input_path + 'mat' + name[index] + '.npy')
        ax = sns.heatmap(mat.T, cmap="RdBu_r", vmax=20000)
        # ax = sns.heatmap(mat.T, annot=True, fmt="d", cmap="RdBu_r")
        ax.invert_yaxis()
        plt.xlabel('block x idx')
        plt.ylabel('block y idx')
        month = 'March'
        plt.title('spatial distribution of population density (' + month + name[index] + ')')
        plt.savefig('tmp.png')
        image_list.append(imageio.imread('tmp.png'))
        plt.pause(0.2)
    imageio.mimsave('pic.gif', image_list, duration=0.2)
    plt.ioff()
    plt.show()


def pin_dist():
    input_path = 'data/March/processed/'
    sns.set_context({"figure.figsize": (12, 5)})
    name = [str(i).zfill(2) for i in range(1, 11)]
    print(len(name))
    mat = np.load(input_path + 'mat' + name[0] + '.npy')
    for index in range(1, 10):
        mat = mat + np.load(input_path + 'mat' + name[index] + '.npy')
    plt.subplot(1, 2, 1)
    ax = sns.heatmap(mat.T / 10, cmap="RdBu_r", vmax=20000)
    ax.invert_yaxis()
    plt.xlabel('block x idx')
    plt.ylabel('block y idx')
    plt.title('ping spatial distribution (Mar. 1-11)')
    print(np.sum(mat / 10))
    print(np.where(mat == np.max(mat)))

    name = [str(i).zfill(2) for i in range(20, 30)]
    print(len(name))
    mat = np.load(input_path + 'mat' + name[0] + '.npy')
    for index in range(1, 10):
        mat = mat + np.load(input_path + 'mat' + name[index] + '.npy')
    plt.subplot(1, 2, 2)
    ax = sns.heatmap(mat.T / 10, cmap="RdBu_r", vmax=20000)
    ax.invert_yaxis()
    plt.xlabel('block x idx')
    plt.ylabel('block y idx')
    plt.title('ping spatial distribution (Mar. 20-30)')
    print(np.sum(mat / 10))
    print(np.where(mat == np.max(mat)))
    plt.show()


def adj_vis():
    input_path = 'data/March/processed/'
    feat = np.loadtxt(input_path + 'adj', delimiter=',')
    print(np.shape(feat))
    sns.set_context({"figure.figsize": (6, 5)})
    ax = sns.heatmap(feat, cmap="RdPu")
    plt.xlabel('Geohash block id')
    plt.ylabel('Geohash block id')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def feat_vis():
    input_path = 'data/March/processed/'
    feat = np.loadtxt(input_path + 'feat', delimiter=',')
    print(np.shape(feat))
    sns.set_context({"figure.figsize": (9, 5)})
    ax = sns.heatmap(feat, cmap="RdPu", vmax=3000)
    plt.xlabel('Time (3h per interval)')
    plt.ylabel('Geohash block id')
    # plt.xticks([])
    # plt.yticks([])
    plt.title('Population distribution during March')
    plt.show()


if __name__ == '__main__':
    pin_dyn_dis()
    # adj_vis()
