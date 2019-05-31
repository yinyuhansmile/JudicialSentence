# -*- coding: utf-8 -*-

import multiprocessing
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedLineDocument
import logging
import time
from hanziconv import HanziConv
import jieba
import json
import data
import sys
import thulac


def trainWord2Vector(sentence_count, vector_dimension, train_count=5 ):

    lines, model_out, vector_out = "sources/splited_words_big.txt", "result/word2vecbig.model", "result/pre_word2vecbig.vector"
    logging.info("开始训练数据")
    sentences = LineSentence(lines)
    # 注意min_count=5表示词频小于3的词 不做计算，，也不会保存到word2vec.vector中
    # workers是训练的进程数，一般等于CPU核数  默认是3
    # sg表示选择的训练算法
    model = Word2Vec(sentences, sg=1, size=vector_dimension, window=10,negative=10,
                     min_count=10, workers=multiprocessing.cpu_count())
    # 多训练几次  使得效果更好
    for i in range(train_count):
        logging.info(i)
        logging.info("训练第%s次"%(i))
        model.train(sentences=sentences, total_examples=sentence_count, epochs=model.iter)
        model_outtmp = "result/tmp/word2vecbig"+str(i)+".model"
        vector_outtmp = "result/tmp/pre_word2vecbig"+str(i)+".vector"
        model.save(model_outtmp)
        model.wv.save_word2vec_format(vector_outtmp)

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(model_out)
    model.wv.save_word2vec_format(vector_out)

def trainWord2Vectorthulac(sentence_count, vector_dimension, train_count=5 ):

    lines, model_out, vector_out = "sources/cut_text_big.txt", "result/word2vecthulacbig.model", "result/pre_word2vecthulacbig.vector"
    logging.info("开始训练数据")
    sentences = LineSentence(lines)
    # 注意min_count=5表示词频小于3的词 不做计算，，也不会保存到word2vec.vector中
    # workers是训练的进程数，一般等于CPU核数  默认是3
    # sg表示选择的训练算法
    model = Word2Vec(sentences, sg=1, size=vector_dimension, window=10,negative=10,
                     min_count=10, workers=multiprocessing.cpu_count())
    # 多训练几次  使得效果更好
    for i in range(train_count):
        logging.info(i)
        logging.info("训练第%s次"%(i))
        model.train(sentences=sentences, total_examples=sentence_count, epochs=model.iter)
        model_outtmp = "result/tmp/word2vecthulacbig"+str(i)+".model"
        vector_outtmp = "result/tmp/pre_word2vecthulacbig"+str(i)+".vector"
        model.save(model_outtmp)
        model.wv.save_word2vec_format(vector_outtmp)

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(model_out)
    model.wv.save_word2vec_format(vector_out)

def trainDoc2Vector(sentence_count, vector_dimension):
    # train and save the model
    sentences = TaggedLineDocument('sources/splited_words.txt')
    model = Doc2Vec(sentences, size=vector_dimension, window=8, min_count=2, workers=multiprocessing.cpu_count())
    model.train(sentences, total_examples=sentence_count, epochs=model.iter)
    model.save('result/doc2vec.model')
    # save vectors
    out = open('result/doc2vec.vector', mode='w+', encoding='utf-8')
    for index in range(0, sentence_count, 1):
        docvec = model.docvecs[index]
        out.write(' '.join(str(f) for f in docvec) + "\n")

    out.close()

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

def loadDataAndSave():
    # 加载过滤词
    filter_read = open("./stopwords/stopwords.txt", mode='r', encoding='utf-8')
    filter_words = set()
    for words in filter_read:
        words = words.strip("\n")
        filter_words.add(words)
    filter_read.close()

    fin = open('./cail_0518/cail2018_big.json', 'r', encoding = 'utf8')
    sentence_write = open("sources/init_sentences.txt", mode='w+', encoding='utf-8')
    words_write = open("sources/splited_words_big.txt", mode='w+', encoding='utf-8')
    f = open("sources/cut_text_big.txt",mode='w+',encoding = 'utf-8')
    word_num = 0
    cut = thulac.thulac(seg_only = True)
    count = 0  
    line = fin.readline()
    alltext = []
    while line :
        d = json.loads(line)
        alltext.append(d['fact'].replace('\r','').replace('\n','').replace('\t','')) 
        line = fin.readline()
#         count += 1
#         if count % 100 == 0:
#             break
    for title in alltext:
        document = HanziConv.toSimplified(title)
        sentence_write.write(document+'\n')
        count += 1
        if count % 2000 == 0:
            print(count)
        cuttext = cut.cut(document, text = True)
        cuttextlist = [] 
        for seg in cuttext.split(' '):
            if seg in filter_words:
                continue
            if seg == " ":
                continue
            cuttextlist.append(seg)
        f.write(' '.join(cuttextlist).strip() + "\n")
        seg_list = jieba.cut(document, cut_all=True)  # seg_list是生成器generator类型
        # 去掉分词中长度为1的词 去掉过滤词
        splited_words = []
        for seg in seg_list:
            if seg in filter_words:
                continue
            if seg == " ":
                continue
            splited_words.append(seg)
            word_num += 1
        words = ' '.join(splited_words) 
        words = words.strip() + "\n"
        words_write.write(words)
    sentence_write.close()
    words_write.close()
    f.close()
    logging.info("词个数为：%s" % word_num)
    return count

import numpy as np
if __name__ == "__main__":
    vector_dimension = 256
    strtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_name = "logs/"+strtime+".txt"
    logging.basicConfig(handlers=[logging.FileHandler(log_name, 'w+', 'utf-8')], format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    stime = time.time()
#     all_nums = loadDataAndSave()
    length = 1710856
    logging.info("text nums :%d"%length)
    trainWord2Vector(sentence_count=length, vector_dimension=vector_dimension)
    trainWord2Vectorthulac(sentence_count=length, vector_dimension=vector_dimension)
#     x = np.zeros((length,10000))
#     for i in range(1000000):
#         print(x.shape)
#     print(x.shape)
    