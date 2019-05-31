#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import gensim
import sys
import logging
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300 , "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "7,7,7", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.6, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

strtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
log_name = "logs/"+strtime+".txt"
logging.basicConfig(handlers=[logging.FileHandler(log_name, 'w+', 'utf-8')], format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# Data Preparation
# ==================================================

# Load data
logging.info("Loading data...")
print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text, y = data_helpers.load_data_and_labels_zh(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
logging.info("Build vocabulary...")
max_document_length = max([len(x.split(" ")) for x in x_text])
max_document_length = 400
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=10)
# x = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_processor.fit(x_text)
all_numtrain = len(x_text)
print("训练集:%s"%all_numtrain)
# print(max_document_length)
logging.info("训练集最大数组长度")
logging.info(max_document_length)
x = np.zeros((all_numtrain,max_document_length))
index = 0
for xrow in x_text:
    a = []
    a.append(xrow)
    x1 = np.array(list(vocab_processor.transform(a)))
    x[index] = x1
    index += 1
del x_text 

xdev_text, ydev = data_helpers.load_data_and_labels_zh_dev(FLAGS.positive_data_file, FLAGS.negative_data_file)
all_numdev = len(xdev_text)
xdev = np.zeros((all_numdev,max_document_length))
print("训练集:%s"%all_numdev)
index = 0
for xrow in xdev_text:
    a = []
    a.append(xrow)
    x1 = np.array(list(vocab_processor.transform(a)))
    xdev[index] = x1
    index += 1
del xdev_text
# x = np.array(list(vocab_processor.fit_transform(x_text)))
# xdev = np.array(list(vocab_processor.fit_transform(xdev_text)))
# x = np.array(list(vocab_processor.transform(x_text)))
# xdev = np.array(list(vocab_processor.transform(xdev_text)))

logging.info(x.shape)
logging.info(xdev.shape)
logging.info(y.shape)
logging.info(ydev.shape)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_train = x[shuffle_indices]
y_train = y[shuffle_indices]
np.random.seed(10)
shuffle_indices_dev = np.random.permutation(np.arange(len(ydev)))
x_dev = xdev[shuffle_indices_dev]
y_dev = ydev[shuffle_indices_dev]

# Split train/test set
# TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# sys.exit()


del x, y, xdev,ydev
print("train shape :%s"%str(x_train.shape))
print("train label shape :%s"%str(y_train.shape))
print("dev shape :%s"%str(x_dev.shape))
print("dev label shape :%s"%str(y_dev.shape))

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
logging.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
logging.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

#========================load pre-trained word2vec=====================================
word_teee = vocab_processor.vocabulary_._mapping
aa = sorted(word_teee.items(),key= lambda item:item[1])
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, encoding='utf-8')
model = gensim.models.KeyedVectors.load_word2vec_format('./result/pre_word2vec.vector',unicode_errors='ignore', encoding='utf-8')
word_vecs = []
for word in aa:
    try:
        word_vecs.append(model[word[0]]) 
    except Exception as err:
        embeddingtmmp = np.random.uniform(-1.0, 1.0, 300)
        word_vecs.append(embeddingtmmp)
        continue
word_array = np.array(word_vecs)

del model,aa,word_teee,word_vecs


# Training
# ==================================================
start_time = time.time()
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: word_array})
        sess.run(tf.global_variables_initializer())
#         print(cnn.W.eval())


        def train_step(x_batch, y_batch):

            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            logging.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            logging.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            print("spend time : %.9f seconds" % ((time.time()-start_time)))
            logging.info("spend time : %.9f seconds" % ((time.time()-start_time)))

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            # sys.exit()
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                logging.info("\nEvaluation:")
                batchesdev = data_helpers.batch_iter_dev(list(zip(x_dev, y_dev)), FLAGS.batch_size)
                for batchdev in batchesdev:
                    x_batch, y_batch = zip(*batchdev) 
                    dev_step(x_batch, y_batch, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                logging.info("Saved model checkpoint to {}\n".format(path))
#             if current_step > 2000 :
#                 break
        
        print("spend final time : %.9f seconds" % ((time.time()-start_time)))
        logging.info("spend final time : %.9f seconds" % ((time.time()-start_time)))
