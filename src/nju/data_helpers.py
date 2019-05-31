# -*- coding: utf-8 -*-

import numpy as np
import re
import sys
from collections import Counter
from _overlapped import NULL
import pickle
from collections import defaultdict
import logging
import mysql.connector
from hanziconv import HanziConv
import jieba
import pandas as pd
import data
import json
lawcount = 183
acccount = 202
timecount = 9 #刑期标签数量

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding = 'utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding = 'utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_zh(positive_data_file='./data/rt-polaritydata/rt-polarity.pos', negative_data_file='./data/rt-polaritydata/rt-polarity.neg'):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from database
    revs = load_dataset()
    x_text = []
    y_text = []
    for rev in revs:
        if(rev["text"] is not None and rev["y"] is not None):
            x_text.append(rev["text"])
            y_text.append(rev["y"])
        else:
            continue
    allnums = len(y_text)
    y = np.zeros((allnums,timecount))
#     set()
#     y = np.zeros((allnums,lawcount))
    index = 0
    for numy in y_text:
        y[index][numy] = 1
        index +=1
        if(index > allnums):
            break
    #print(y.shape)
#     print(x_text[0])
    print(y[0])
#     y=pd.get_dummies(y_text).values
#     print(y)
#     rowtext = len(y_text)
#     i = y.shape[1]
#     while (i < lawcount):
#         coloumtext = np.zeros((rowtext,1))
# #         print(coloumtext)
#         y = np.column_stack((y,coloumtext))
#         i +=1
#     print(y)
    
    return [x_text, y]

def load_data_and_labels_zh_dev(positive_data_file='./data/rt-polaritydata/rt-polarity.pos', negative_data_file='./data/rt-polaritydata/rt-polarity.neg'):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from database
    revs = load_dataset_dev()
    x_text = []
    y_text = []
    for rev in revs:
#         print(rev)
        if(rev["text"] is not None and rev["y"] is not None):
            x_text.append(rev["text"])
            y_text.append(rev["y"])
        else:
            continue
    allnums = len(y_text)
    #y = np.zeros((allnums,acccount))
#     y = np.zeros((allnums,lawcount))
    y = np.zeros((allnums,timecount))
    index = 0
    for numy in y_text:
        y[index][numy] = 1
        index +=1
        if(index > allnums):
            break
    print(y.shape)
    return [x_text, y]


def load_data_and_labels_zh_test(positive_data_file='./data/rt-polaritydata/rt-polarity.pos', negative_data_file='./data/rt-polaritydata/rt-polarity.neg'):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from database
    revs = load_dataset_test()
    x_text = []
    y_text = []
    for rev in revs:
        if(rev["text"] is not None and rev["y"] is not None):
            x_text.append(rev["text"])
            y_text.append(rev["y"])
        else:
            continue
#     print(y)
  
    return [x_text, y_text]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
#     print(data_size)
#     sys.exit()
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
def batch_iter_dev(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
#     print(data_size)
#     sys.exit()
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]
            
def read_trainData(path):
    fin = open(path, 'r', encoding = 'utf8')
    
    alltext = []
    
    accu_label = []
    law_label = []
    time_label = []

    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])
        accu_label.append(data.getlabel(d, 'accu'))
        law_label.append(data.getlabel(d, 'law'))
        time_label.append(data.getlabel(d, 'time'))
        line = fin.readline()
    fin.close()
    return alltext, accu_label, law_label, time_label

def load_dataset(cv=10):
    # 加载过滤词
    filter_read = open("./stopwords/stopwords.txt", mode='r', encoding='utf-8')
    filter_words = set()
    write_flag = 0
    for words in filter_read:
        words = words.strip("\n")
        if words in filter_words:
            write_flag = 1
            logging.info("过滤词典中有重复词:%s" % words)
        filter_words.add(words)
    filter_read.close()
    # 出现了重复词 则要更新词表
    if write_flag == 1:
        filter_write = open("./stopwords/stopwords.txt", mode='w+', encoding='utf-8')
        for word in filter_words:
            filter_write.write(word+"\n")
        filter_write.close()

    values, accu_label, law_label, time_label = read_trainData('./cail_0518/data_train.json')
    print("-------------------------训练集--------------------------")
    revs = []
    vocab = defaultdict(float)
    index  = 0
    count = 0
    for title in values:
        #y = accu_label[index]
#         y = law_label[index]
        y = time_label[index]
        index +=1
        document = HanziConv.toSimplified(title)
        seg_list = jieba.cut(document, cut_all=True)  # seg_list是生成器generator类型
#         print(' '.join(seg_list))
        # 去掉分词中长度为1的词 去掉过滤词
        splited_words = []
        for seg in seg_list:
            if seg in filter_words:
                continue
            if len(seg)<1:
                continue
            splited_words.append(seg)
#         if(len(splited_words)>820):
#             count +=1
#             continue
#         words = ' '.join(splited_words) 
        orig_rev = " ".join(splited_words).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {"y": y,
                  "text": orig_rev,
                  "num_words": len(orig_rev.split())}
        revs.append(datum)
        if(index > 10000):
            break
    print("---------------丢弃的数量---------------")
    print(count)
    return revs

def load_dataset_dev(cv=10):
    # 加载过滤词
    filter_read = open("./stopwords/stopwords.txt", mode='r', encoding='utf-8')
    filter_words = set()
    for words in filter_read:
        words = words.strip("\n")
        filter_words.add(words)
    filter_read.close()

    values, accu_label, law_label, time_label = read_trainData('./cail_0518/data_valid.json')
#     print(len(values))
#     print(len(law_label))
    print("-----------------------验证集----------------------------")
    revs = []
    vocab = defaultdict(float)
    index  = 0
    count = 0 
    for title in values:
        #y = accu_label[index]
        y = time_label[index]
#         y = law_label[index]
        index +=1
        document = HanziConv.toSimplified(title)
        seg_list = jieba.cut(document, cut_all=True)  # seg_list是生成器generator类型
        # 去掉分词中长度为1的词 去掉过滤词
        splited_words = []
        for seg in seg_list:
            if seg in filter_words:
                continue
            if len(seg)<1:
                continue
            splited_words.append(seg)
#         if(len(splited_words)>800):
#             count +=1
#             continue
#         words = ' '.join(splited_words) 
        orig_rev = " ".join(splited_words).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {"y": y,
                  "text": orig_rev,
                  "num_words": len(orig_rev.split())}
        revs.append(datum)
        if(index > 200):
            break
#     print("---------------丢弃的数量---------------")
#     print(count)
#         print(datum)
    return revs

def load_dataset_test(cv=10):
    # 加载过滤词
    filter_read = open("./stopwords/stopwords.txt", mode='r', encoding='utf-8')
    filter_words = set()
    for words in filter_read:
        words = words.strip("\n")
        filter_words.add(words)
    filter_read.close()

    values, accu_label, law_label, time_label = read_trainData('./cail_0518/data_test.json')
    print(len(values))
    print(len(accu_label))
    print("-----------------------测试集----------------------------")
    revs = []
    index  = 0
    for title in values:
        #y = accu_label[index]
        y = time_label[index]
#         y = law_label[index]
        index +=1
        document = HanziConv.toSimplified(title)
        seg_list = jieba.cut(document, cut_all=True)  # seg_list是生成器generator类型
        # 去掉分词中长度为1的词 去掉过滤词
        splited_words = []
        for seg in seg_list:
            if seg in filter_words:
                continue
            if len(seg)<1:
                continue
            splited_words.append(seg) 
        orig_rev = " ".join(splited_words).lower()
        words = set(orig_rev.split())
        datum  = {"y": y,
                  "text": orig_rev,
                  "num_words": len(orig_rev.split())}
        revs.append(datum)
#         print(datum)
        if(index > 100):
            break
    return revs

import gensim
import logging


if __name__=="__main__": 
    load_dataset_dev()
    load_data_and_labels_zh()
    load_data_and_labels_zh_dev()
#     load_dataset()
#     load_dataset()
#     sys.exit()
#     load_data_and_labels_zh()
#     sys.exit() 
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#     model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', unicode_errors='ignore', binary=True,encoding='utf-8')#cn.skipgram.bin  
#     print(model.wv.most_similar('man')) 
#     sys.exit() 
 
