import json
import thulac
from sklearn.externals import joblib
from .TfidfWord2vec import TfidfEmbeddingVectorizer
import gensim


class Predictor(object):
	def __init__(self):
		self.tfidf = joblib.load('predictor/model/tfidf11small.model')
		self.law = joblib.load('predictor/model/lawword2vecsmall.model')
		self.accu = joblib.load('predictor/model/accuword2vecsmall.model')
		self.time = joblib.load('predictor/model/timeword2vecsmall.model')
		model = gensim.models.KeyedVectors.load_word2vec_format('predictor/model/pre_word2vecthulac.vector',unicode_errors='ignore', encoding='utf-8')
		w2v = model.wv
		self.tfidfword2vec = TfidfEmbeddingVectorizer(w2v,self.tfidf,300)
		self.tfidfword2vec.fit()
		self.batch_size = 100000
		
		self.cut = thulac.thulac(seg_only = True)

	def predict_law(self, vec):
		y = self.law.predict(vec)
		return [y[0] + 1]
	
	def predict_accu(self, vec):
		y = self.accu.predict(vec)
		return [y[0] + 1]
	
	def predict_time(self, vec):

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
		
	def predict(self, content):
		anslist = []
		for xtext in content:
			fact = self.cut.cut(xtext, text = True)
			vec1 = self.tfidfword2vec.transform([fact])
			ans = {}
			ans['accusation'] = self.predict_accu(vec1)
			ans['articles'] = self.predict_law(vec1)
			ans['imprisonment'] = self.predict_time(vec1)
			anslist.append(ans)
			print(ans)
		return anslist

		 
