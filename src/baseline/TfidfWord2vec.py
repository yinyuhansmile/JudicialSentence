#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.externals import joblib
import numpy as np
import gensim
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, tfidf, dim=300):
        self.word2vec = word2vec
        self.word2weight = None
        self.tfidf = tfidf
        self.dim= dim
#         if len(word2vec)>0:
#             self.dim= dim
#         else:
#             self.dim=0
        
    def fit(self):
#         tfidf = TfidfVectorizer(analyzer=lambda x: x)
#         tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(self.tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, self.tfidf.idf_[i]) for w, i in self.tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


