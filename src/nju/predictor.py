import json
import thulac
from sklearn.externals import joblib

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import jieba
from hanziconv import HanziConv

class Predictor(object):
	def __init__(self):
		# Eval Parameters
		tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
		tf.flags.DEFINE_string("checkpoint_dir_law", "./model1/checkpoints", "Law checkpoint directory from training run")
		tf.flags.DEFINE_string("checkpoint_dir_accu", "./model1/checkpoints", "Accu checkpoint directory from training run")
		tf.flags.DEFINE_string("checkpoint_dir_time", "./model1/checkpoints", "Time checkpoint directory from training run")
		tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
		
		# Misc Parameters
		tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
		tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
		FLAGS = tf.flags.FLAGS
		FLAGS._parse_flags()
		
		self.tfidf = joblib.load('predictor/model/tfidf.model')
		self.law = joblib.load('predictor/model/law.model')
		self.accu = joblib.load('predictor/model/accu.model')
		self.time = joblib.load('predictor/model/time.model')
		
		self.batch_size = 64
		self.cut = jieba.cut()
		self.vocab_path = ""
		self.vocab_procerror = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
		self.graph = tf.Graph()

#     def predict_law(self, vec):
#     	y = self.law.predict(vec)
#     	return [y[0] + 1]
	
	def predict_law_svm(self, vec):
		y = self.law.predict(vec)
		return [y[0] + 1]
	
	def predict_accu_svm(self, vec):
		y = self.accu.predict(vec)
		return [y[0] + 1]
	
	def predict_time_svm(self, vec):

		y = self.time.predict(vec)[0]
		
		#返回每一个罪名区间的中位数		
		if y == 0:
			return -2
		if y == 1:
			return -1
		if y == 2:
			return 120
		if y == 3:
			return 102
		if y == 4:
			return 72
		if y == 5:
			return 48
		if y == 6:
			return 30
		if y == 7:
			return 18
		else:
			return 6
		
	def get_stopwords(self):
		filter_read = open("./stopwords/stopwords.txt", mode='r', encoding='utf-8')
		filter_words = set()
		for words in filter_read:
			words = words.strip("\n")
			filter_words.add(words)
		filter_read.close()
		return filter_words
		
	def predict(self, content):
		filter_words = self.get_stopwords()
		x_raw = []
		for text in content:
			document = HanziConv.toSimplified(text)
			seg_list = self.cut(document, cut_all=True)
			splited_words = []
			for seg in seg_list:
				if seg in filter_words:
					continue
				if len(seg)<1:
					continue
				splited_words.append(seg)
			orig_rev = " ".join(splited_words).lower()
            x_raw.append(orig_rev)
		x_test = np.array(list(self.vocab_procerror.transform(x_raw)))

		vec = self.tfidf.transform([fact])
		ans = {}

		ans['accusation'] = self.predict_accu(vec)
		ans['articles'] = self.predict_law(vec)
		ans['imprisonment'] = self.predict_time(vec)
		
		print(ans)
		return [ans]

		 
