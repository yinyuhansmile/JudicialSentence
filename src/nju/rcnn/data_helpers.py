#!usr/bin/python 
# -*- coding:utf-8 -*- 

"""
Construct a Data generator.
"""
import numpy as np
from tqdm import tqdm
import os


class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=True)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]
                
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
    
    
def to_categorical_accu(label):
    """把所有的label id 转为 0，1形式。
    Args:
        label: n_sample 个 lists, 文书的罪名标签。每个list对应一个文书，label个数不定。
    return:
        y: ndarray, shape=(sample， n_class)， 其中 n_class = 202.
    Example:
     >>> y_batch = to_categorical(label_batch)
     >>> print(y_batch.shape)
     >>> (10, 202)
    """
    n_sample = len(label)
    y = np.zeros(shape=(n_sample, 202))
    for i in range(n_sample):
        topic_index = label[i]
        y[i, topic_index] = 1
    return y

def to_categorical_law(label):
    """把所有的label id 转为 0，1形式。
    Args:
        label: n_sample 个 lists, 文书的法条标签。每个list对应一个文书，label个数不定。
    return:
        y: ndarray, shape=(sample， n_class)， 其中 n_class = 202.
    Example:
     >>> y_batch = to_categorical(label_batch)
     >>> print(y_batch.shape)
     >>> (10, 202)
    """
    n_sample = len(label)
    y = np.zeros(shape=(n_sample, 183))
    for i in range(n_sample):
        topic_index = label[i]
        y[i, topic_index] = 1
    return y

def to_categorical_time(label):
    """把所有的label id 转为 0，1形式。
    Args:
        label: n_sample 个 lists, 文书的刑期标签。每个list对应一个文书，label个数不定。
    return:
        y: ndarray, shape=(sample， n_class)， 其中 n_class = 66.
    Example:
     >>> y_batch = to_categorical(label_batch)
     >>> print(y_batch.shape)
     >>> (10, 1)
    """
    n_sample = len(label)
    y = np.zeros(shape=(n_sample, 66))
    for i in range(n_sample):
        topic_index = label[i]
        y[i, topic_index] = 1
    return y

def to_categorical_time_value(label):
    """把所有的label id 转为 0，1形式。
    Args:
        label: n_sample 个 lists, 文书的刑期。每个list对应一个文书，label个数不定。
    return:
        y: ndarray, shape=(sample， n_class)， 其中 n_class = 1.
    Example:
     >>> y_batch = to_categorical(label_batch)
     >>> print(y_batch.shape)
     >>> (10, 1)
    """
    n_sample = len(label)
    y = np.zeros(shape=(n_sample, 1))
    for i in range(n_sample):
        y[i] = label[i]
    return y


def train_batch(X, y, batch_path, batch_size=128):
    """对训练集打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    batch_num = 0
    for start in tqdm(range(0, sample_num, batch_size)):
        end = min(start + batch_size, sample_num)
        batch_name = batch_path+ str(batch_num) + '.npz'
        X_batch = X[start:end]
        y_batch = y[start:end]
        np.savez(batch_name, X=X_batch, y=y_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num+1))
    
def train_batch_predict(X, y, batch_path, batch_num, batch_size=128):
    """对训练集打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    batch_name = batch_path+ str(batch_num) + '.npz'
    X_batch = X
    y_batch = y
    np.savez(batch_name, X=X_batch, y=y_batch)


def eval_batch(X, batch_path, batch_size=128):
    """对测试数据打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    print('sample_num=%d' % sample_num)
    batch_num = 0
    for start in tqdm(range(0, sample_num, batch_size)):
        end = min(start + batch_size, sample_num)
        batch_name = batch_path + str(batch_num) + '.npy'
        X_batch = X[start:end]
        np.save(batch_name, X_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num+1))
    
# new_batch = np.load('../data/predictbatch/accu/0' + '.npz')
# X_batch = new_batch['X']
# y_batch = new_batch['y']
# print((X_batch))
# print(y_batch)
import time
import sys
def get_batch(data_path, batch_id):
    """get a batch from data_path"""
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    return [X_batch, y_batch]
# a= [1,2,3]
# print(a[2:])
# batchpath = '..\\data\\jieba-data\\data_train\\train_time_label.npy'
# a = np.load(batchpath)
# print(list(a))
# count1 = 0
# count2 = 0
# for i in list(a) :
#     if(i<37 and i ):
#         if(i%3 != 0):
#             print(i)
#             count1+=1
#         if(i%6 != 0):
#             print(i)
#             count2+=1
# print(count1)
# print(count2)

# for i in range(8902):
#     starttime = time.time()
# #     print(time.time())
#     get_batch(batchpath, i)
#     print(time.time()-starttime)
#     sys.exit()