# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import shutil
import time
from nju.fasttext.network import Settings
from nju.fasttext.network import Fasttext
from nju.fasttext.judger import Judger
from nju.fasttext.data_helpers import train_batch_predict
import logging
import datetime

# sys.path.append('../..')
from nju.fasttext.data_helpers import to_categorical_accu
from nju.fasttext.data_helpers import to_categorical_law
from nju.fasttext.data_helpers import to_categorical_time


strtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
# log_name = "../logs/"+strtime+"_fasttext_old.txt"
# log_name = "../logs/"+strtime+"law_fasttext_newaug.txt"
log_name = "../logs/"+strtime+"accu_fasttext_new1aug.txt"
logging.basicConfig(handlers=[logging.FileHandler(log_name, 'w+', 'utf-8')], format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

flags = tf.flags
flags.DEFINE_bool('is_retrain', False, 'if is_retrain is true, not rebuild the summary')
flags.DEFINE_integer('max_epoch', 1, 'update the embedding after max_epoch, default: 1')
flags.DEFINE_integer('max_max_epoch', 10, 'all training epoches, default: 6')
flags.DEFINE_float('lr', 1e-3, 'initial learning rate, default: 1e-3')
flags.DEFINE_float('decay_rate', 0.65, 'decay rate, default: 0.65')
flags.DEFINE_float('keep_prob', 0.5, 'keep_prob for training, default: 0.5')
# 正式
flags.DEFINE_integer('decay_step', 8000, 'decay_step, default: 15000')
flags.DEFINE_integer('valid_step', 6000, 'valid_step, default: 10000')
flags.DEFINE_float('last_score', 0.5, 'if valid_score > last_score, save new model. default: 0.75')

# 测试
# flags.DEFINE_integer('decay_step', 1000, 'decay_step, default: 1000')
# flags.DEFINE_integer('valid_step', 500, 'valid_step, default: 500')
# flags.DEFINE_float('last_f1', 0.10, 'if valid_f1 > last_f1, save new model. default: 0.10')
FLAGS = flags.FLAGS

lr = FLAGS.lr
last_score = FLAGS.last_score
settings = Settings()
# jieb = settings.jieba_len
summary_path = settings.summary_path
ckpt_path = settings.ckpt_path
model_path = ckpt_path + 'model.ckpt'

batch_path = '../data/predictbatch/new1/fasttext/'
embedding_path = '../data/word_embedding.npy'
words_path = '../data/sr_word2id.pkl'
# data_train_path = '../data/jieba-data/data_train/batch/accu/'
# data_valid_path = '../data/jieba-data/data_valid/batch/accu/'
data_train_path = '../data/new1/jieba-data/aug/data_train/batch/accu/'
data_valid_path = '../data/new1/jieba-data/data_test/batch/accu/'
# data_valid_path = '../data/new1/jieba-data/aug/data_valid/batch/accu/'
# data_train_path = '../data/new1/jieba-data/aug/data_train/batch/law/'
# # data_valid_path = '../data/new1/jieba-data/data_test/batch/law/'
# data_valid_path = '../data/new1/jieba-data/aug/data_valid/batch/law/'
tr_batches = os.listdir(data_train_path)  # batch 文件名列表
va_batches = os.listdir(data_valid_path)
n_tr_batches = len(tr_batches)
n_va_batches = len(va_batches)


accusation_path = '../cail_0518/accu.txt'
law_path = '../cail_0518/law.txt'
judger = Judger(accusation_path,law_path)

# 测试
# n_tr_batches = 1000
# n_va_batches = 50


def get_batch_old(data_path, batch_id):
    """get a batch from data_path"""
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    X1_batch = X_batch[:, :1]
    X2_batch = X_batch[:, 1:]
    return [X1_batch, X2_batch, y_batch]

def get_batch(data_path, batch_id):
    """get a batch from data_path"""
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    return [X_batch, y_batch]


def valid_epoch(data_path, sess, model):
    """Test on the valid data."""
    va_batches = os.listdir(data_path)
    n_va_batches = len(va_batches)
    _costs = 0.0
    marked_labels_list = list()
    y_pred_list = list()
#     n_va_batches = 10
    for i in range(n_va_batches):
        [X_batch, y_batch1] = get_batch(data_path, i)
        marked_labels_list.extend(y_batch1)
        y_batch = to_categorical_accu(y_batch1)#罪名标签转换
#         y_batch = to_categorical_law(y_batch1) #法条标签转换
#         y_batch = to_categorical_time(y_batch1) #刑期标签转换
        _batch_size = len(y_batch)
        fetches = [model.loss, model.y_pred, model.accuracy]
        feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch,
                     model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
        _cost, predict_labels, _accuracy = sess.run(fetches, feed_dict)
        y_pred_list.extend(predict_labels)
#         train_batch_predict(predict_labels,y_batch1, batch_path+'time/', i, batch_size=128)
#         train_batch_predict(predict_labels,y_batch1, batch_path+'law/', i, batch_size=128)
        train_batch_predict(predict_labels,y_batch1, batch_path+'accu/', i, batch_size=128)
        dictscore = {}
    maxscore = 0
    thresholdlist = [-0.80,-0.78,-0.76,0]
    for threshold in thresholdlist:   
        predict_labels_list = list()  # 所有的预测结果 
        for label in y_pred_list:
            xitem = np.argwhere(label>threshold).flatten()
            if(len(xitem)>0):
                predict_labels_list.append(xitem)
            else:
                predict_labels_list.append(label.argsort()[-1:-2:-1]) 
#         print(len(predict_labels_list))
#         print(len(marked_labels_list))
        predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
        score = judger.get_taskaccu_score(predict_label_and_marked_label_list) #罪名预测分数
#         score = judger.get_tasklaw_score(predict_label_and_marked_label_list) #法条预测分数
#         score = judger.get_tasktime_score(predict_label_and_marked_label_list) #刑期回归预测分数
#         score = judger.get_tasktime_class_score(predict_label_and_marked_label_list) #刑期分类预测分数
        dictscore[threshold] = score
        if(score > maxscore):
            maxscore = score
#         predict_labels_list.extend(list(predict_labels.flatten())) #刑期回归预测
#         predict_labels = map(lambda label: label.argsort()[-1:-2:-1], predict_labels)  #刑期分类预测
#         predict_labels_list.extend(predict_labels)
    mean_cost = _costs / n_va_batches
    return mean_cost, dictscore, maxscore


def train_epoch(data_path, sess, model, train_fetches, valid_fetches, train_writer, test_writer):
    global last_score
    global lr
    time0 = time.time()
    batch_indexs = np.random.permutation(n_tr_batches)  # shuffle the training data
    for batch in tqdm(range(n_tr_batches)):
        global_step = sess.run(model.global_step)
        if 0 == (global_step + 1) % FLAGS.valid_step:
            valid_cost, dictscore, score = valid_epoch(data_valid_path, sess, model)
            print('\n')
            print('Global_step=%d: valid cost=%g; score=%g, time=%g s' % (
                global_step, valid_cost, score, time.time() - time0))
            logging.info('Global_step=%d: valid cost=%g; score=%g, time=%g s' % (
                global_step, valid_cost, score, time.time() - time0))
            for threshold,thresholdscore in dictscore.items():
                print('threshold=%g: score=%g ' % (threshold, thresholdscore))
                logging.info('threshold=%g: score=%g ' % ( threshold, thresholdscore))
            time0 = time.time()
            if score > last_score:
                last_score = score
                saving_path = model.saver.save(sess, model_path, global_step+1)
                print('\n')
                print('saved new model to %s ' % saving_path)
                logging.info('saved new model to %s ' % saving_path)
        # training
        batch_id = batch_indexs[batch]
        [X_batch, y_batch] = get_batch(data_train_path, batch_id)
        y_batch = to_categorical_accu(y_batch) #罪名标签转换
#         y_batch = to_categorical_law(y_batch) #法条标签转换
#         y_batch = to_categorical_time(y_batch) #刑期标签转换
        _batch_size = len(y_batch)
        feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch,
                     model.batch_size: _batch_size, model.tst: False, model.keep_prob: FLAGS.keep_prob}
        summary, _cost, _accuracy, _, _ = sess.run(train_fetches, feed_dict)  # the cost is the mean cost of one batch
        time_str = datetime.datetime.now().isoformat()
#         print("{}: step {}, loss {:g}, acc {:g}".format(time_str, global_step, _cost, _accuracy))
        if 0 == (global_step + 1) % 200:
            logging.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, global_step, _cost, _accuracy))
        # valid per 500 steps
        if 0 == (global_step + 1) % 500:
            train_writer.add_summary(summary, global_step)
            batch_id = np.random.randint(0, n_va_batches)  # 随机选一个验证batch
            [X_batch, y_batch] = get_batch(data_valid_path, batch_id)
            y_batch = to_categorical_accu(y_batch) #罪名标签转换
#             y_batch = to_categorical_law(y_batch) #法条标签转换
#             y_batch = to_categorical_time(y_batch) #刑期标签转换
            _batch_size = len(y_batch)
            feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch,
                         model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
            summary, _cost, _accuracy = sess.run(valid_fetches, feed_dict)
            time_str = datetime.datetime.now().isoformat()
#             print("valid: {}: step {}, loss {:g}, acc {:g}".format(time_str, global_step, _cost, _accuracy))
            logging.info("valid: {}: step {}, loss {:g}, acc {:g}".format(time_str, global_step, _cost, _accuracy))
            test_writer.add_summary(summary, global_step)


def main(_):
    global ckpt_path
    global last_score
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    elif not FLAGS.is_retrain:  # 重新训练本模型，删除以前的 summary
        shutil.rmtree(summary_path)
        os.makedirs(summary_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    print('1.Loading data...')
    logging.info('1.Loading data...')
    W_embedding = np.load(embedding_path)
    print('training sample_num = %d' % n_tr_batches)
    logging.info('training sample_num = %d' % n_tr_batches)
    print('valid sample_num = %d' % n_va_batches)
    logging.info('valid sample_num = %d' % n_va_batches)

    # Initial or restore the model
    print('2.Building model...')
    logging.info('2.Building model...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Fasttext(W_embedding, settings)
        logging.info('word embedding path %s'%embedding_path)
        logging.info('text length = %d'%settings.jieba_len)
        logging.info('text length = %d'%settings.thulac_len)
        logging.info('train task class = %d'%settings.n_class)
        logging.info('model name %s'%settings.model_name)
#         logging.info('filter size = %s'%str(settings.filter_sizes))
#         logging.info('per filter size = %d'%settings.n_filter)
        with tf.variable_scope('training_ops') as vs:
            learning_rate = tf.train.exponential_decay(FLAGS.lr, model.global_step, FLAGS.decay_step,
                                                   FLAGS.decay_rate, staircase=True)
            # two optimizer: op1, update embedding; op2, do not update embedding.
            with tf.variable_scope('Optimizer1'):
                tvars1 = tf.trainable_variables()
                grads1 = tf.gradients(model.loss, tvars1)
                optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op1 = optimizer1.apply_gradients(zip(grads1, tvars1),
                                                   global_step=model.global_step)
            with tf.variable_scope('Optimizer2'):
                tvars2 = [tvar for tvar in tvars1 if 'embedding' not in tvar.name]
                grads2 = tf.gradients(model.loss, tvars2)
                optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op2 = optimizer2.apply_gradients(zip(grads2, tvars2),
                                                   global_step=model.global_step)
            update_op = tf.group(*model.update_emas)
            merged = tf.summary.merge_all()  # summary
            train_writer = tf.summary.FileWriter(summary_path + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(summary_path + 'test')
            training_ops = [v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]

        # 如果已经保存过模型，导入上次的模型
        if os.path.exists(ckpt_path + "checkpoint"):
            print("Restoring Variables from Checkpoint...")
            logging.info("Restoring Variables from Checkpoint...")
            model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            valid_cost, dictscore, score = valid_epoch(data_valid_path, sess, model)
            print('Global_step=%d: valid cost=%g; score=%g' % (
                sess.run(model.global_step), valid_cost,score))
            logging.info('Global_step=%d: valid cost=%g; score=%g' % (
                sess.run(model.global_step), valid_cost, score))
            for threshold,thresholdscore in dictscore.items():
                print('threshold=%g: score=%g ' % (threshold, thresholdscore))
                logging.info('threshold=%g: score=%g ' % ( threshold, thresholdscore))
            sess.run(tf.variables_initializer(training_ops))
            train_op2 = train_op1
        else:
            print('Initializing Variables...')
            logging.info('Initializing Variables...')
            sess.run(tf.global_variables_initializer())

        print('3.Begin training...')
        logging.info('3.Begin training...')
        print('max_epoch=%d, max_max_epoch=%d' % (FLAGS.max_epoch, FLAGS.max_max_epoch))
        logging.info('max_epoch=%d, max_max_epoch=%d' % (FLAGS.max_epoch, FLAGS.max_max_epoch))
        train_op = train_op2
        for epoch in range(FLAGS.max_max_epoch):
            global_step = sess.run(model.global_step)
            print('Global step %d, lr=%g' % (global_step, sess.run(learning_rate)))
            logging.info('Global step %d, lr=%g' % (global_step, sess.run(learning_rate)))
            if epoch == FLAGS.max_epoch:  # update the embedding
                train_op = train_op1

            train_fetches = [merged, model.loss, model.accuracy, train_op, update_op]
            valid_fetches = [merged, model.loss, model.accuracy]
            train_epoch(data_train_path, sess, model, train_fetches, valid_fetches, train_writer, test_writer)
        # 最后再做一次验证
        valid_cost, dictscore, score = valid_epoch(data_valid_path, sess, model)
        print('Global_step=%d: valid cost=%g; score=%g' % (
            sess.run(model.global_step), valid_cost,score))
        logging.info('Global_step=%d: valid cost=%g; score=%g' % (
            sess.run(model.global_step), valid_cost, score))
        for threshold,thresholdscore in dictscore.items():
            print('threshold=%g: score=%g ' % (threshold, thresholdscore))
            logging.info('threshold=%g: score=%g ' % ( threshold, thresholdscore))
        if score > last_score:  # save the better model
            saving_path = model.saver.save(sess, model_path, sess.run(model.global_step)+1)
            print('saved new model to %s ' % saving_path)
            logging.info('saved new model to %s ' % saving_path)


if __name__ == '__main__':
    tf.app.run()
