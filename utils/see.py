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


def tf_basic():
    sess = tf.Session()
    # gt = tf.constant([2, 0])
    # predict_logits = tf.constant([[2.0, -1.0, 3.0], [1.0, 0.0, -0.5]])
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=gt,
    #     logits=predict_logits
    # )
    # print(sess.run(loss))

    r1 = tf.random_uniform([10, 1, 59, 2])
    r2 = tf.ones([10, 1, 59], dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=r2, logits=r1)
    loss = tf.reduce_sum(loss, [0, 1, 2])
    print(sess.run(loss))

    r3 = tf.random_uniform([10, 1, 59, 1], dtype=tf.float32)
    r4 = tf.constant(418.0, shape=r3.shape)
    cmp = tf.greater(r3, r4)
    cmp = tf.cast(cmp, dtype=tf.int32)
    print(sess.run(cmp))


if __name__ == '__main__':
    tf_basic()
    # feat_stat()
    # ex = np.loadtxt('dataset/apr_X_314.csv', delimiter`=',')
    # print(ex.shape)
    # adj_vis('dataset/featPingSDenseT.csv')