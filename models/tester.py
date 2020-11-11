# @Time     : Jan. 10, 2019 17:52
# @Author   : Veritas YIN
# @FileName : tester.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from numpy.lib.npyio import load
from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation, class_evaluation
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time


def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]


def class_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        pred = sess.run(y_pred,feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
        if isinstance(pred, list):
            pred = np.array(pred[0])
        pred_list.append(pred)
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array, pred_array.shape[0]


def model_inference_cls(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, max_va_val, max_val):
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
    label_val, label_test = inputs.get_label('val'), inputs.get_label('test')

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    y_val, len_val = class_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
    val_acc, val_f1, val_prec, val_recall = class_evaluation(label_val[0:len_val, step_idx + n_his, :, :], y_val)
    print(y_val.reshape(-1)[:10])
    # evl_val, evl_0, evl_f1, evl_precision, evl_recall = class_evaluation(label_val[0:len_val, step_idx + n_his, :, :], y_val)

    # compare with copy prediction
    cp_val = x_val[0:len_val, step_idx + n_his - 1, :, 0] * x_stats['std']  + x_stats['mean']
    cp_y = np.digitize(cp_val, x_stats['bin']) - 1
    cp_y[cp_y >= len(x_stats['bin']) - 1] = len(x_stats['bin']) - 2
    cp_acc, cp_f1, cp_prec, cp_recall = class_evaluation(label_val[0:len_val, step_idx + n_his, :, :], cp_y)

    print(f"Val Acc: {val_acc:.3%} F1 {val_f1:.3%} Precision: {val_prec:.3%} Recall: {val_recall:.3%}")
    print(f"Copy Acc: {cp_acc:.3%} F1 {cp_f1:.3%} Precision: {cp_prec:.3%} Recall: {cp_recall:.3%}")

    chks = val_acc > max_va_val
    if chks:
        max_va_val = val_acc
        y_pred, len_pred = class_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        test_acc, test_f1, test_prec, test_recall = class_evaluation(label_test[0:len_pred, step_idx + n_his, :, :], y_pred)
        max_val = test_acc
    return max_va_val, max_val


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)

    # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
    chks = evl_val < min_va_val
    # update the metric on test set, if model's performance got improved on the validation.
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
        min_val = evl_pred

    return min_va_val, min_val


def model_test_cls(inputs, batch_size, n_his, n_pred, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
        label_test = inputs.get_label('test')

        y_test, len_test = class_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, 0)
        val_acc, val_f1, val_prec, val_recall = class_evaluation(label_test[0:len_test, n_his, :, :], y_test)

        cp_val = x_test[0:len_test, n_his - 1, :, 0]* x_stats['std']  + x_stats['mean']
        cp_y = np.digitize(cp_val, x_stats['bin']) - 1
        cp_y[cp_y >= len(x_stats['bin']) - 1] = len(x_stats['bin']) - 2
        cp_acc, cp_f1, cp_prec, cp_recall = class_evaluation(label_test[0:len_test, n_his, :, :], cp_y)

        print(f"Val Acc: {val_acc:.3%} F1 {val_f1:.3%} Precision: {val_prec:.3%} Recall: {val_recall:.3%}")
        print(f"Copy Acc: {cp_acc:.3%} F1 {cp_f1:.3%} Precision: {cp_prec:.3%} Recall: {cp_recall:.3%}")

        print(f'Model Test Time {time.time() - start_time:.3f}s')

    print('Testing model finished!')


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)

        for ix in tmp_idx:
            if inf_mode == 'sep':
                te = evl[ix: ix + 3]
            elif inf_mode == 'merge':
                te = evl[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')

def dataErrorMap(inputs, batch_size, n_his, n_pred, load_path='./output/models/'):
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        x_train, x_stats = inputs.get_data('train'), inputs.get_stats()
        label_train = inputs.get_label('train')
        label_train = label_train[:, n_his, :, 0]

        y_train, len_train = class_pred(test_sess, pred, x_train, batch_size, n_his, n_pred, 0)
        print(y_train.shape)
        print(label_train.shape)
        train_diff = y_train - label_train

        x_val = inputs.get_data('val')
        label_val = inputs.get_label('val')
        label_val = label_val[:, n_his, :, 0]
        y_val, len_val = class_pred(test_sess, pred, x_val, batch_size, n_his, n_pred, 0)
        val_diff = y_val - label_val

        print(f'Error Map Generation Time {time.time() - start_time:.3f}s')
        np.savetxt(pjoin(load_path, "trainError.csv"), train_diff.transpose(), delimiter=',')
        np.savetxt(pjoin(load_path, "valError.csv"), val_diff.transpose(), delimiter=',')
        np.savetxt(pjoin(load_path, "Bin.csv"), x_stats['bin'], delimiter=',')
    print('Testing model finished!')

