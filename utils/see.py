import numpy as np
import tensorflow as tf
# import seaborn as sns
# import matplotlib.pyplot as plt


def adj_vis(fp):
    feat = np.loadtxt(fp, delimiter=',')
    # feat = feat[480:, :]
    print(np.shape(feat))
    # np.savetxt('dataset/part.csv', feat, delimiter=',', fmt='%d')
    sns.set_context({"figure.figsize": (6, 5)})
    ax = sns.heatmap(feat.T, cmap="RdPu")
    plt.xlabel('T')
    plt.ylabel('Geohash block id')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def feat_stat(fp='dataset/part.csv'):
    feat = np.loadtxt(fp, delimiter=',')
    m = np.median(feat)
    print(m)
    cmp = feat > m


def tf_test():
    sess = tf.Session()
    r1 = tf.random_uniform([10, 1])


if __name__ == '__main__':
    tf_test()
    # feat_stat()
    # ex = np.loadtxt('dataset/apr_X_314.csv', delimiter`=',')
    # print(ex.shape)
    # adj_vis('dataset/featPingSDenseT.csv')