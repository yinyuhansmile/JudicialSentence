from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
from predictor import data
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import pickle
import thulac
import sys
from .TfidfWord2vec import TfidfEmbeddingVectorizer
import gensim

dim = 5000


def cut_text(alltext):
	filename = 'cut_text_train.txt'
# 	f = open(filename,mode='a',encoding = 'utf-8')
# 	count = 0	
# 	cut = thulac.thulac(seg_only = True)
# 	train_text = []
# 	for text in alltext:
# 		count += 1
# 		if count % 2000 == 0:
# 			print(count)
# 		cuttext = cut.cut(text, text = True)
# 		train_text.append(cuttext)
# 		f.write(cuttext+'\n')
# 	f.close()
	f = open(filename,mode='r',encoding = 'utf-8')
	train_text = []
	while True:
		line = f.readline()
		if not line:
			break 
		train_text.append(line.replace('\n', '').strip())
	f.close()
	return train_text


def train_tfidf(train_data):
	tfidf = TFIDF(
			min_df = 5,
			max_features = dim,
			ngram_range = (1, 1),
			use_idf = 1,
			smooth_idf = 1
			)
	tfidf.fit(train_data)
	
	return tfidf



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


def train_SVC(vec, label):
	SVC = LinearSVC()
	SVC.fit(vec, label)
# 	clf1 = DecisionTreeClassifier(max_depth=4)
# 	clf2 = KNeighborsClassifier(n_neighbors=7)
# 	eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', SVC)], voting='hard') 
# 	eclf.fit(vec, label)

	return SVC


if __name__ == '__main__':
	print('reading...')
	alltext, accu_label, law_label, time_label = read_trainData('data_train.json')
	print('cut text...')
	train_data = cut_text(alltext)
	print('train tfidf...')
# 	tfidf = joblib.load('predictor/model/tfidf11.model')
	tfidf = train_tfidf(train_data)
	joblib.dump(tfidf, 'predictor/model/tfidf11small.model')
	
	print('train tfidfword2vec...')
	model = gensim.models.KeyedVectors.load_word2vec_format('predictor/model/pre_word2vecthulac.vector',unicode_errors='ignore', encoding='utf-8')
	w2v = model.wv
	tfidfword2vec = TfidfEmbeddingVectorizer(w2v,tfidf,300)
	tfidfword2vec.fit()
	
# 	vec = tfidf.transform(train_data,accu_label)
	vec = tfidfword2vec.transform(train_data)
# 	print(len(vec))
# 	print(vec[0])
	
	del tfidfword2vec,model,tfidf,alltext,train_data
	
	print('accu SVC')
	accu = train_SVC(vec, accu_label)
	joblib.dump(accu, 'predictor/model/accuword2vecsmall.model')
	print('law SVC')
	law = train_SVC(vec, law_label)
	joblib.dump(law, 'predictor/model/lawword2vecsmall.model')
	print('time SVC')
	time = train_SVC(vec, time_label)
	joblib.dump(time, 'predictor/model/timeword2vecsmall.model')
	
	print('saved model')