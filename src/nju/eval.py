#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys
import pickle

from tensorflow.contrib.tensorboard.plugins import projector

def visualize(vocab, output_path):
    embeddingarray = pickle.load(open("D:/embedding.array","rb"))
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(vocab), 300))
    print(len(vocab))
    print(embeddingarray.shape)
    
    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for word in vocab:
            placeholder[word[1]] = embeddingarray[word[1]] 
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word[0]).encode('utf-8') + b'\n')
    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))



# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/accu/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print(FLAGS.checkpoint_dir)
FLAGS._parse_flags()
print("\nParameters:")
print("what the ")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels_zh_test(FLAGS.positive_data_file, FLAGS.negative_data_file)
#     y_test = np.argmax(y_test, axis=1)
    print(y_test)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]
# 
# # Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
# word_teee = vocab_processor.vocabulary_._mapping
# aa = sorted(word_teee.items(),key= lambda item:item[1])
# 
# 
# del word_teee,aa
# print("\nEvaluating...\n")

# Evaluation
# ==================================================
print('---------------')
print(FLAGS.checkpoint_dir)
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
# print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format('E:/Lawresearchcup/src/nju/runs/accu/checkpoints/model-10000'))
        saver.restore(sess, 'E:/Lawresearchcup/src/nju/runs/accu/checkpoints/model-10000')

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        
#         W = graph.get_operation_by_name("embedding/W").outputs[0]
#         embedding = W.eval()
#         pickle.dump(embedding,open("D:/embedding.array", "wb"))
#         sys.exit()

        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
#         print(predictions)

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
#         print(batches)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    print(all_predictions)
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
    
    

